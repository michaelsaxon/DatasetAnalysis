import torch
import transformers as tf

trainer = tf.Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset
)

