import torch
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, List, Any

# =============================================================================
# 1. 설정 및 데이터 클래스
# =============================================================================
@dataclass
class EngineConfig:
    audio_input_tokens: int = 4   
    text_output_tokens: int = 2   
    audio_output_tokens: int = 4  
    silence_token_id: int = 151646 
    
    system_prompt_text: str = (
        "<|im_start|>system\n"
        "You are a funny comedian performing a stand-up comedy show using Qwen3-Omni.\n"
        "<|im_end|>\n"
    )

# =============================================================================
# 2. 로직 클래스
# =============================================================================
class Qwen3DuplexLogic:
    def __init__(self, model):
        self.model = model
        self.device = model.device
        
        self.thinker_device = model.thinker.device
        self.talker_device = model.talker.device
        self.code2wav_device = model.code2wav.device
        
        self.talker_config = model.config.talker_config
        # ★ [수정] 모델 설정에서 Codec Layer 개수 확인 (기본값 16)
        self.num_quantizers = getattr(self.talker_config, "num_quantizers", 16)
        
        # Audio Tower Dtype 확인
        try:
            self.audio_dtype = model.thinker.audio_tower.conv2d1.weight.dtype
        except:
            self.audio_dtype = model.dtype

    @torch.no_grad()
    def thinker_step(
        self,
        input_ids: Optional[torch.Tensor],
        input_features: Optional[torch.Tensor], # ★ 수정: Audio Features 받음
        feature_attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[List],
        step_idx: int
    ):
        # [Multi-GPU Safety]
        if input_ids is not None and input_ids.device != self.thinker_device:
            input_ids = input_ids.to(self.thinker_device)
        if input_features is not None:
            if input_features.device != self.thinker_device:
                input_features = input_features.to(self.thinker_device)
            # Dtype 맞춤
            input_features = input_features.to(dtype=self.audio_dtype)
        if feature_attention_mask is not None and feature_attention_mask.device != self.thinker_device:
            feature_attention_mask = feature_attention_mask.to(self.thinker_device)

        # RoPE Position IDs 생성
        # Audio 입력 시: feature 길이만큼 / Text 입력 시: text 길이만큼
        if input_ids is None and input_features is not None:
            # 더미 토큰 (예: 패딩 토큰이나 <|audio|> 토큰 등)
            # 여기선 단순히 길이 1짜리 텐서를 만들고 무시되길 기대하거나,
            # 모델이 input_features가 있으면 input_ids를 무시하도록 설계되었는지 확인 필요.
            # 가장 안전한 건: input_ids에 <|audio|> 토큰 하나 넣어주는 것.
            
            # 151646 등 특수 토큰 사용? 그냥 0번 토큰 사용
            input_ids = torch.tensor([[0]], device=self.thinker_device)

        position_ids = torch.tensor([[step_idx]], device=self.thinker_device)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # Thinker Forward
        # 성공했던 코드처럼 input_features를 직접 넘김
        outputs = self.model.thinker(
            input_ids=input_ids,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=True
        )
        
        return outputs

    @torch.no_grad()
    def talker_step(
        self,
        thinker_hidden: torch.Tensor,
        past_key_values: Optional[List],
        step_idx: int,
        input_ids: Optional[torch.Tensor] = None
    ):
        if thinker_hidden.device != self.talker_device:
            thinker_hidden = thinker_hidden.to(self.talker_device)
        
        if input_ids is None:
             input_ids = torch.tensor([[self.model.config.talker_config.codec_bos_id]], device=self.talker_device)
        else:
             input_ids = input_ids.to(self.talker_device)

        conditioned_hidden = self.model.talker.text_projection(thinker_hidden)
        audio_embed = self.model.talker.model.get_input_embeddings()(input_ids)
        talker_inputs_embeds = audio_embed + conditioned_hidden
        
        position_ids = torch.tensor([[step_idx]], device=self.talker_device)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        talker_out = self.model.talker.model(
            inputs_embeds=talker_inputs_embeds,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True
        )
        
        logits = self.model.talker.codec_head(talker_out.last_hidden_state[:, -1, :])
        layer0_code = logits.argmax(dim=-1, keepdim=True)
        
        last_id_hidden = self.model.talker.get_input_embeddings()(layer0_code)
        past_hidden = talker_out.last_hidden_state[:, -1:]
        predictor_input = torch.cat((past_hidden, last_id_hidden), dim=1)
        
        # ★ [수정] 전체 16개 중 1개(Layer0)는 이미 나왔으므로 15개를 더 생성해야 함
        needed_tokens = self.num_quantizers - 1
        
        predictor_out = self.model.talker.code_predictor.generate(
            inputs_embeds=predictor_input,
            max_new_tokens=needed_tokens, # 7 -> needed_tokens (15)로 변경
            do_sample=False
        )
        
        full_audio_codes = torch.cat([layer0_code, predictor_out], dim=1)
        return full_audio_codes, talker_out.past_key_values

    @torch.no_grad()
    def decode_audio(self, audio_codes: torch.Tensor) -> bytes:
        if audio_codes.device != self.code2wav_device:
            audio_codes = audio_codes.to(self.code2wav_device)
        if audio_codes.dim() == 2:
            audio_codes = audio_codes.unsqueeze(-1)
            
        wav_tensor = self.model.code2wav(audio_codes)
        wav_np = wav_tensor.cpu().float().numpy()
        wav_int16 = (wav_np * 32767).astype(np.int16)
        return wav_int16.tobytes()

# =============================================================================
# 3. 엔진 클래스
# =============================================================================
class Qwen3OmniFullDuplexEngine:
    def __init__(self, model, tokenizer, config: EngineConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.logic = Qwen3DuplexLogic(model)
        
        # Queues
        self.input_queue = queue.Queue()   
        self.hidden_queue = queue.Queue()  
        self.output_queue = queue.Queue()  
        
        # States
        self.thinker_kv_cache = None
        self.talker_kv_cache = None
        self.text_history_ids = None 
        self.last_talker_token = None
        
        self.thinker_step_count = 0
        self.talker_step_count = 0
        
        self.is_running = False
        self.t_thinker = None
        self.t_talker = None

        self._initialize_context()

    def _initialize_context(self):
        print("⚡ [Engine] Initializing...")
        initial_ids = self.tokenizer(
            self.cfg.system_prompt_text, 
            return_tensors="pt", 
            add_special_tokens=False
        ).input_ids.to(self.logic.thinker_device)
        
        # Talker Init
        codec_bos = self.model.config.talker_config.codec_bos_id
        self.last_talker_token = torch.tensor([[codec_bos]], device=self.logic.talker_device)

        # Prefill Thinker (Text Only)
        with torch.no_grad():
            # Init 시에는 Feature 없이 Text만
            out = self.logic.thinker_step(
                input_ids=initial_ids,
                input_features=None,
                feature_attention_mask=None,
                past_key_values=None,
                step_idx=0
            )
            self.thinker_kv_cache = out.past_key_values
            self.thinker_step_count = initial_ids.shape[1]
            
        print("✅ [Engine] Ready.")

    def _thinker_loop(self):
        print("🧠 [Thinker Thread] Running...")
        while self.is_running:
            try:
                # ★ Feature(Mel Spec)를 받음
                audio_features = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with torch.no_grad():
                # [Step 1] Audio Feature 입력 -> Thinker Forward
                # Feature Mask 생성 (성공 코드 참조)
                time_len = audio_features.shape[2]
                feature_mask = torch.ones((1, time_len), device=self.logic.thinker_device, dtype=torch.long)

                # ★ [수정 핵심] input_ids가 None이면 에러가 나므로, 
                # 오디오만 처리할 때도 형식을 맞춰줘야 함.
                # Qwen3-Omni는 오디오 처리 시 input_ids를 안 쓸 수도 있지만,
                # transformers 구현체에 따라 input_ids를 요구할 수 있음.
                # 여기서는 input_ids=None으로 호출하되, logic.py에서 처리하도록 위임했으나
                # 에러가 났으므로 더미 input_ids를 넣어줌.
                
                # 하지만 더미를 넣으면 텍스트가 섞일 수 있으니,
                # 가장 안전한 방법: logic.py의 thinker_step 수정
                
                thinker_out = self.logic.thinker_step(
                    input_ids=None, 
                    input_features=audio_features,
                    feature_attention_mask=feature_mask,
                    past_key_values=self.thinker_kv_cache,
                    step_idx=self.thinker_step_count
                )
                self.thinker_kv_cache = thinker_out.past_key_values
                
                # Step Count 증가 (성공 코드에서는 오디오 처리 후 +1만 했음. 정확히는 +time_len 이지만 
                # Qwen3 스트리밍 특성상 압축된 토큰 수만큼 증가시키는게 맞음. 
                # 일단 4토큰(0.32s)에 대해 1스텝 증가로 가정하고 진행)
                # (만약 자네 성공 코드가 1스텝만 증가시켰다면 1이 맞음)
                self.thinker_step_count += 4 # 4 audio tokens 입력되었으므로

                # [Step 2] Text Generation
                # 첫 토큰 예측 (오디오 통과 결과에서)
                next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                
                current_turn_hiddens = []
                
                # 첫 번째 예측 토큰 처리
                if next_token.item() == self.cfg.silence_token_id:
                    pass # Silence면 넘어감
                else:
                    # 첫 토큰에 대한 Hidden State 저장
                    current_turn_hiddens.append(thinker_out.hidden_states[-1])
                    
                    # 2번째 토큰부터 생성 (설정된 갯수만큼)
                    for _ in range(self.cfg.text_output_tokens - 1):
                        thinker_out = self.logic.thinker_step(
                            input_ids=next_token,
                            input_features=None,
                            feature_attention_mask=None,
                            past_key_values=self.thinker_kv_cache,
                            step_idx=self.thinker_step_count
                        )
                        self.thinker_kv_cache = thinker_out.past_key_values
                        self.thinker_step_count += 1
                        
                        next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        
                        if next_token.item() == self.cfg.silence_token_id:
                            break
                        
                        current_turn_hiddens.append(thinker_out.hidden_states[-1])

                # Talker Queue에 넣기
                if len(current_turn_hiddens) > 0:
                    stacked_hidden = torch.cat(current_turn_hiddens, dim=1)
                    self.hidden_queue.put(stacked_hidden)

    def _talker_loop(self):
        print("👄 [Talker Thread] Running...")
        while self.is_running:
            try:
                source_hidden = self.hidden_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            with torch.no_grad():
                num_hiddens = source_hidden.shape[1]
                # Text 1개당 Audio 2개 (2:4 비율)
                ratio = self.cfg.audio_output_tokens // self.cfg.text_output_tokens
                
                for i in range(num_hiddens):
                    one_hidden = source_hidden[:, i:i+1, :]
                    for _ in range(ratio):
                        codes, new_kv = self.logic.talker_step(
                            thinker_hidden=one_hidden,
                            past_key_values=self.talker_kv_cache,
                            step_idx=self.talker_step_count,
                            input_ids=self.last_talker_token
                        )
                        self.talker_kv_cache = new_kv
                        self.talker_step_count += 1
                        self.last_talker_token = codes[:, 0:1] # Layer 0 Code
                        
                        wav_bytes = self.logic.decode_audio(codes)
                        self.output_queue.put(wav_bytes)

    def start(self):
        if self.is_running: return
        self.is_running = True
        self.t_thinker = threading.Thread(target=self._thinker_loop, daemon=True)
        self.t_talker = threading.Thread(target=self._talker_loop, daemon=True)
        self.t_thinker.start()
        self.t_talker.start()
        print("🚀 Engine Threads Started.")

    def stop(self):
        self.is_running = False
        if self.t_thinker: self.t_thinker.join()
        if self.t_talker: self.t_talker.join()
        print("🛑 Engine Threads Stopped.")

    def push_audio(self, audio_features: torch.Tensor):
        self.input_queue.put(audio_features)

    def get_audio_output(self) -> Optional[bytes]:
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None