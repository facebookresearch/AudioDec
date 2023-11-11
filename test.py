from AudioDec.utils.audiodec import AudioDec as AudioDecModel, assign_model

if model_sr == "24khz":
    self.sr, encoder_checkpoint, decoder_checkpoint = assign_model('libritts_v1')
elif model_sr == "48khz":
    self.sr, encoder_checkpoint, decoder_checkpoint = assign_model('vctk_v1')