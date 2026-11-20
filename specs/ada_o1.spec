# ADA Unified Compiler + WSP Writer Specification Sheet
num_neurons = 256
substrate_dim = 128
tile_size = 16
vq_codebook_size = 8192
topology = KDA,KDA,KDA,MLA,KDA,KDA,KDA,MLA

kda_chunk_size = 64
kda_dplr_rank = 4
kda_state_dim = 128

mla_kv_latent_dim = 64
mla_num_heads = 8
mla_head_dim = 64

workspace_slots = 16
workspace_max_iters = 8
workspace_halting_tolerance = 1e-6

neurons_per_capability = 64
decoder_train_tokens = 250000
decoder_eval_tokens = 50000
decoder_lambda = 0.1
