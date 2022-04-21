from pathlib import Path

import tensorflow as tf

from infercode.client.infercode_trainer import InferCodeTrainer
from infercode.settings import DATA_DIR, SAVED_MODEL_PATH

# TF1 to TF2
var_names_map = {
    'dense/kernel': 'infer_code_model/dense1/kernel',
    'dense_1/kernel': 'infer_code_model/dense2/kernel',
    'inputs/node_token_embeddings': 'infer_code_model/node_token_embeddings',
    'inputs/w_attention': 'infer_code_model/code_encoder/w_attention',
    'inputs/node_type_embeddings': 'infer_code_model/node_type_embeddings',
    'inputs/w_t_0': 'infer_code_model/tb_conv_layer/w_t',
    'inputs/w_l_0': 'infer_code_model/tb_conv_layer/w_l',
    'inputs/w_r_0': 'infer_code_model/tb_conv_layer/w_r',
    'inputs/b_conv_0': 'infer_code_model/tb_conv_layer/b_conv',
}
expand_dims_vars = {'inputs/w_t_0', 'inputs/w_l_0', 'inputs/w_r_0', 'inputs/b_conv_0'}


def print_checkpoint(save_path):
    reader = tf.train.load_checkpoint(save_path)
    shapes = reader.get_variable_to_shape_map()
    dtypes = reader.get_variable_to_dtype_map()
    print(f"Checkpoint at '{save_path}':")
    for key in shapes:
        print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}, "
              f"value={reader.get_tensor(key)})")


def convert_tf1_to_tf2(checkpoint_path, output_prefix):
    """Converts a TF1 checkpoint to TF2.

    To load the converted checkpoint, you must build a dictionary that maps
    variable names to variable objects.
    ```
    ckpt = tf.train.Checkpoint(vars={name: variable})
    ckpt.restore(converted_ckpt_path)

      ```

      Args:
        checkpoint_path: Path to the TF1 checkpoint.
        output_prefix: Path prefix to the converted checkpoint.

      Returns:
        Path to the converted checkpoint.
      """
    vars = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    dtypes = reader.get_variable_to_dtype_map()
    with tf.compat.v1.Session() as sess:
        for key in dtypes.keys():
            if key not in var_names_map:
                continue
            new_name = var_names_map[key]
            if key in expand_dims_vars:
                vars[new_name] = tf.Variable(tf.expand_dims(reader.get_tensor(key), axis=0), name=new_name)
            else:
                vars[new_name] = tf.Variable(reader.get_tensor(key), name=new_name)
        saver = tf.compat.v1.train.Saver()
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.save(sess, output_prefix)


if __name__ == '__main__':
    home = Path.home()
    checkpoint_path_old = home / '.infercode_data' / 'model_checkpoint' / 'universal_model'
    checkpoint_path_new = str(DATA_DIR / 'ckpt' / 'infer_code_model')
    convert_tf1_to_tf2(checkpoint_path_old, checkpoint_path_new)
    print(tf.train.list_variables(checkpoint_path_old))
    print(tf.train.list_variables(checkpoint_path_new))
    infercode_trainer = InferCodeTrainer()
    infercode_model = infercode_trainer.model
    ckpt = tf.train.Checkpoint(infercode_model)
    ckpt.restore(checkpoint_path_new)
    infercode_model.save(SAVED_MODEL_PATH)
