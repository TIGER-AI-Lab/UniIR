from transformers.models.t5 import T5Config
from transformers.models.clip import CLIPConfig


conf_clip = CLIPConfig()
conf_clip.projection_dim = 512
conf_clip.text_config.max_length = 128
conf_clip.text_config.max_position_embeddings = 128

conf_t5 = T5Config()    
conf_t5.num_layers = 2
conf_t5.num_decoder_layers = 2
conf_t5.num_heads = 12
conf_t5.d_model = 512
conf_t5.d_kv = 64

conf_t5_vit_large = T5Config()
conf_t5_vit_large.num_layers = 2
conf_t5_vit_large.num_decoder_layers = 2
conf_t5_vit_large.num_heads = 12
conf_t5_vit_large.d_model = 768
conf_t5_vit_large.d_kv = 64
