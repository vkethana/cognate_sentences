from make_data import get_ratio_from_sentence

def get_score(samples, **kwargs):
  return [10 * get_ratio_from_sentence(sentence) for sentence in samples]

if __name__ == '__main__':
  config.train.batch_size = 1
# freeze all transformer layers
  config.model.num_layers_unfrozen = 0
# maximum sample length, prompts or samples longer than that will be truncated
  config.train.seq_length = 128

# micro batch size for sampling (specific for PPO)
  config.method.chunk_size = 1
# use an additional Q-head (specific for ILQL)
  config.method.two_qs = False

  model_name = "projecte-aina/aguila-7b"
  trainer = trlx.train(model_name, reward_fn=lambda samples, **kwargs: [sample.count('cats') for sample in samples])
  trainer.save_pretrained('output/')

