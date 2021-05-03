def defaultconf():
    cnp_conf = {
        'input_dim':512,
        'control_dim':32,
        'output_dim':512,
        'spatial_dim':512,
        'temporal_dim':512,
        'temporal_n_layers':6,
        'temporal_n_heads':8,
        'temporal_d_k':64,
        'temporal_d_v':64,
        'temporal_hidden_dim':2048,
        'decoder_dim':512,
        'decoder_n_layers':6,
        'decoder_n_heads':8,
        'decoder_d_k':64,
        'decoder_d_v':64,
        'decoder_hidden_dim':2048,
        'fusion_n_heads':8,
        'fusion_d_k':64,
        'fusion_d_v':64,
        'fusion_hidden_dim':2048,
        'struct_dim':312,
        'logit_struct_periph_dim':512,
        'max_seq_len':500,
        'output_embedding_dim':300,
        'dropout':0.1,
        'use_s_decoder':False,
        'use_p_decoder':False,
        'inject_at_logits':False,
        'inject_at_encoder':False,
        'inject_after_encoder':False,
        'inject_at_decoder':False,
        'temp_fusion_attn_type':'default',
        'spat_fusion_attn_type':'default',
        'convex_gate':False,
        'pooling':False,
        'no_logit_struct_peripheral':False}
    perph_conf = {
        'german_language_input_vocab': 25000,
        'german_language_input_embed': 300,
        'english_language_input_vocab': 25000,
        'english_language_input_embed': 300,
        'english_language_output_vocab': 25000,
        'german_language_output_vocab': 25000,
        'dropout': 0.1 ,
        'vqa_output_vocab':3500,
        'hmdb_output_classes':52,
        'penn_output_classes':48,
        'birds_output_classes':200,
        'mm_output_classes':2,
        'image_struct_clf_output_classes':2,
        'struct_dropout': 0.1,
        'struct_temp_dropout': 0.1,
        'struct_spat_dropout': 0.1,
        'unfreeze': {'img': False, 'en': False, 'de': False, 'struct_logits': False, 
            'struct_temp': False, 'struct_spat': False}
    }

    domains = ['ENGLISH','GERMAN','IMAGE', 'STRUCT', 'STRUCT_SPAT', 'STRUCT_TEMP']

    return cnp_conf, perph_conf, domains

def vqa_struct_config():
    cnp_conf = {
        'input_dim':512,
        'control_dim':32,
        'output_dim':512,
        'spatial_dim':512,
        'temporal_dim':512,
        'temporal_n_layers':6,
        'temporal_n_heads':8,
        'temporal_d_k':64,
        'temporal_d_v':64,
        'temporal_hidden_dim':2048,
        'decoder_dim':512,
        'decoder_n_layers':6,
        'decoder_n_heads':8,
        'decoder_d_k':64,
        'decoder_d_v':64,
        'decoder_hidden_dim':2048,
        'fusion_n_heads':1,
        'fusion_d_k':512,
        'fusion_d_v':512,
        'fusion_hidden_dim':2048,
        'struct_dim':128,
        'logit_struct_periph_dim':512,
        'max_seq_len':500,
        'output_embedding_dim':300,
        'dropout':0.1,
        'use_s_decoder':False,
        'use_p_decoder':False,
        'inject_at_logits':False,
        'inject_at_encoder':False,
        'inject_after_encoder':False,
        'inject_at_decoder':False,
        'temp_fusion_attn_type':'default',
        'spat_fusion_attn_type':'default',
        'convex_gate':False,
        'pooling':False,
        'no_logit_struct_peripheral':False}
    perph_conf = {
        'german_language_input_vocab': 25000,
        'german_language_input_embed': 300,
        'english_language_input_vocab': 25000,
        'english_language_input_embed': 300,
        'english_language_output_vocab': 25000,
        'german_language_output_vocab': 25000,
        'dropout': 0.1 ,
        'vqa_output_vocab':3500,
        'hmdb_output_classes':52,
        'penn_output_classes':48,
        'birds_output_classes':200,
        'mm_output_classes':2,
        'image_struct_clf_output_classes':2,
        'struct_dropout': 0.1,
        'struct_temp_dropout': 0.1,
        'struct_spat_dropout': 0.1,
        'unfreeze': {'img': False, 'en': False, 'de': False, 'struct_logits': False, 
            'struct_temp': False, 'struct_spat': False}
    }

    domains = ['ENGLISH','GERMAN','IMAGE', 'STRUCT', 'STRUCT_SPAT', 'STRUCT_TEMP']

    return cnp_conf, perph_conf, domains

def timeline_conf():
    """
    The confurigation for timeline project

    """
    cnp_conf = {
        'input_dim':64, #512,
        'control_dim':32,
        'output_dim':64, #512,
        'spatial_dim':64, #512,
        'temporal_dim':64, #512,
        'temporal_n_layers': 2, #6,
        'temporal_n_heads': 2, #8,
        'temporal_d_k':32, #64,
        'temporal_d_v':32, #64,
        'temporal_hidden_dim': 128, #2048,
        'decoder_dim': 64, #512,
        'decoder_n_layers': 2, #6,
        'decoder_n_heads': 2, #8,
        'decoder_d_k': 32, #64,
        'decoder_d_v': 32, #64,
        'decoder_hidden_dim': 128, #2048,
        'fusion_n_heads': 2, #8,
        'fusion_d_k': 32, #64,
        'fusion_d_v': 32, #64,
        'fusion_hidden_dim': 128, #2048,
        'struct_dim':312,
        'logit_struct_periph_dim':512,
        'max_seq_len':512,
        'output_embedding_dim': 50, #300,
        'dropout':0.5,
        'use_s_decoder':False,
        'use_p_decoder':False,
        'inject_at_logits':False,
        'inject_at_encoder':False,
        'inject_after_encoder':False,
        'inject_at_decoder':False,
        'temp_fusion_attn_type':'default',
        'spat_fusion_attn_type':'default',
        'convex_gate':False,
        'pooling':False,
        'no_logit_struct_peripheral':False}
    perph_conf = {
        'german_language_input_vocab': 25000,
        'german_language_input_embed': 300,
        'english_language_input_vocab': 25000,
        'english_language_input_embed': 300,
        'english_language_output_vocab': 25000,
        'german_language_output_vocab': 25000,
        'dropout': 0.25,
        'vqa_output_vocab':3500,
        'hmdb_output_classes':52,
        'penn_output_classes':48,
        'birds_output_classes':200,
        'mm_output_classes':2,
        'image_struct_clf_output_classes':2,
        'struct_dropout': 0.1,
        'struct_temp_dropout': 0.1,
        'struct_spat_dropout': 0.1,
        'unfreeze': {'img': False, 'en': False, 'de': False, 'struct_logits': False, 
            'struct_temp': False, 'struct_spat': False}
    }

    domains = ['ENGLISH','GERMAN','IMAGE', 'STRUCT', 'STRUCT_SPAT', 'STRUCT_TEMP']

    return cnp_conf, perph_conf, domains

def timeline_conf_small():
    """
    The confurigation for timeline project

    """
    cnp_conf = {
        'input_dim':32, #512,
        'control_dim':16,
        'output_dim':32, #512,
        'spatial_dim':32, #512,
        'temporal_dim':32, #512,
        'temporal_n_layers': 2, #6,
        'temporal_n_heads': 2, #8,
        'temporal_d_k':16, #64,
        'temporal_d_v':16, #64,
        'temporal_hidden_dim': 64, #2048,
        'decoder_dim': 32, #512,
        'decoder_n_layers': 2, #6,
        'decoder_n_heads': 2, #8,
        'decoder_d_k': 16, #64,
        'decoder_d_v': 16, #64,
        'decoder_hidden_dim': 64, #2048,
        'fusion_n_heads': 2, #8,
        'fusion_d_k': 16, #64,
        'fusion_d_v': 16, #64,
        'fusion_hidden_dim': 64, #2048,
        'struct_dim':312,
        'logit_struct_periph_dim':512,
        'max_seq_len':512,
        'output_embedding_dim':32, #300,
        'dropout':0.1,
        'use_s_decoder':False,
        'use_p_decoder':False,
        'inject_at_logits':False,
        'inject_at_encoder':False,
        'inject_after_encoder':False,
        'inject_at_decoder':False,
        'temp_fusion_attn_type':'default',
        'spat_fusion_attn_type':'default',
        'convex_gate':False,
        'pooling':False,
        'no_logit_struct_peripheral':False}
    perph_conf = {
        'german_language_input_vocab': 25000,
        'german_language_input_embed': 300,
        'english_language_input_vocab': 25000,
        'english_language_input_embed': 300,
        'english_language_output_vocab': 25000,
        'german_language_output_vocab': 25000,
        'dropout': 0.1,
        'vqa_output_vocab':3500,
        'hmdb_output_classes':52,
        'penn_output_classes':48,
        'birds_output_classes':200,
        'mm_output_classes':2,
        'image_struct_clf_output_classes':2,
        'struct_dropout': 0.1,
        'struct_temp_dropout': 0.1,
        'struct_spat_dropout': 0.1,
        'unfreeze': {'img': False, 'en': False, 'de': False, 'struct_logits': False, 
            'struct_temp': False, 'struct_spat': False}
    }

    domains = ['ENGLISH','GERMAN','IMAGE', 'STRUCT', 'STRUCT_SPAT', 'STRUCT_TEMP']

    return cnp_conf, perph_conf, domains


def birds_config():
    """
    The confurigation for timeline project

    """
    cnp_conf = {
        'input_dim':256,
        'control_dim':32,
        'output_dim':256,
        'spatial_dim':256,
        'temporal_dim':256,
        'temporal_n_layers': 3, #6,
        'temporal_n_heads': 4, #8,
        'temporal_d_k':64,
        'temporal_d_v':64,
        'temporal_hidden_dim': 1024,
        'decoder_dim': 256,
        'decoder_n_layers': 3,
        'decoder_n_heads': 4,
        'decoder_d_k': 64,
        'decoder_d_v': 64,
        'decoder_hidden_dim': 1024,
        'fusion_n_heads': 4,
        'fusion_d_k': 64,
        'fusion_d_v': 64,
        'fusion_hidden_dim': 1024,
        'struct_dim':312,
        'logit_struct_periph_dim':512,
        'max_seq_len':512,
        'output_embedding_dim': 300,
        'dropout':0.1,
        'use_s_decoder':False,
        'use_p_decoder':False,
        'inject_at_logits':False,
        'inject_at_encoder':False,
        'inject_after_encoder':False,
        'inject_at_decoder':False,
        'temp_fusion_attn_type':'default',
        'spat_fusion_attn_type':'default',
        'convex_gate':False,
        'pooling':False,
        'no_logit_struct_peripheral':False}
    perph_conf = {
        'german_language_input_vocab': 25000,
        'german_language_input_embed': 300,
        'english_language_input_vocab': 25000,
        'english_language_input_embed': 300,
        'english_language_output_vocab': 25000,
        'german_language_output_vocab': 25000,
        'dropout': 0.1 ,
        'vqa_output_vocab':3500,
        'hmdb_output_classes':52,
        'penn_output_classes':48,
        'birds_output_classes':200,
        'mm_output_classes':2,
        'image_struct_clf_output_classes':2,
        'struct_dropout': 0.1,
        'struct_temp_dropout': 0.1,
        'struct_spat_dropout': 0.1,
        'unfreeze': {'img': False, 'en': False, 'de': False, 'struct_logits': False, 
            'struct_temp': False, 'struct_spat': False}
    }

    domains = ['ENGLISH','GERMAN','IMAGE', 'STRUCT', 'STRUCT_SPAT', 'STRUCT_TEMP']

    return cnp_conf, perph_conf, domains