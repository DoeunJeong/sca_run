import torch
import numpy as np
import asyncio
import time
from dataclasses import dataclass
from typing import Optional, List, Any

# Moshi 스타일 로거 임포트
try:
    from .client_utils import log, get_logger
except ImportError:
    def log(level, msg): print(f"[{level.upper()}] {msg}")
    def get_logger(): 
        class FallbackLogger:
            def print_token(self, t, color=None): print(t, end="", flush=True)
        return FallbackLogger()

# =============================================================================
# 1. 설정 및 데이터 클래스
# =============================================================================
@dataclass
class EngineConfig:
    audio_input_tokens: int = 4   
    text_output_tokens: int = 2   
    audio_output_tokens: int = 4  
    silence_token_id: int = 151646 
    audio_token_id: int = 151675

    system_prompt_text: str = (
        "<|im_start|>system\n"
        "You are a funny comedian performing a stand-up comedy show using Qwen3-Omni.\n"
        "<|im_end|>\n"
    )

# =============================================================================
# 2. 로직 클래스 (Stateless Tensor Operations)
# =============================================================================

class Qwen3DuplexLogic:
    def __init__(self, model):
        self.model = model
        self.device = model.device # 대표 디바이스 (보통 cuda:0)
        
        # [수정] 각 모듈의 실제 디바이스 위치를 파악하여 저장
        # 모델이 분산되어 있을 경우 thinker_device와 talker_device가 다름
        if hasattr(model, "thinker"):
            self.thinker_device = model.thinker.device
        else:
            self.thinker_device = self.device

        if hasattr(model, "talker"):
            # Talker의 첫 번째 파라미터 위치를 기준으로 잡음
            self.talker_device = next(model.talker.parameters()).device
        else:
            self.talker_device = self.device
            
        if hasattr(model, "code2wav"):
            self.code2wav_device = next(model.code2wav.parameters()).device
        else:
            self.code2wav_device = self.device

        self.talker_config = model.config.talker_config
        self.num_quantizers = getattr(self.talker_config, "num_quantizers", 16)
        
        try:
            self.audio_dtype = model.thinker.audio_tower.conv2d1.weight.dtype
        except:
            self.audio_dtype = model.dtype

    def _calc_audio_token_count(self, input_lengths):
        """
        [공식 코드 Line 99 기반]
        Audio Encoder의 Convolution 및 Window Chunking을 고려한 정확한 토큰 수 계산
        """
        # 1. 윈도우(100프레임) 단위로 처리되지 않는 나머지 부분 계산
        input_lengths_leave = input_lengths % 100
        
        # 2. Convolution Layer 시뮬레이션 (Stride 2, Kernel 3, Padding 1 적용)
        #    Layer 1
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        #    Layer 2 & 3 (복합 공식 적용)
        output_lengths_leave = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1
        
        # 3. 전체 토큰 수 = (윈도우 개수 * 13) + 나머지 부분 토큰 수
        #    * Qwen3-Omni는 100프레임 윈도우 하나당 13개의 토큰을 생성합니다.
        total_output_lengths = output_lengths_leave + (input_lengths // 100) * 13
        
        return int(total_output_lengths)

    @torch.no_grad()
    def thinker_step(self, input_ids, input_features, feature_attention_mask, past_key_values, step_idx=None):
        """
        Thinker Step: Audio or Text Input -> Next Token Prediction
        """
        # ★ [FIX] 초기화: inputs_embeds 변수가 모든 분기에서 정의되도록 함
        target_device = self.thinker_device
        inputs_embeds = None
        
        try:
            # 1. Audio Input Case
            if input_features is not None:
                if input_features.device != target_device:
                    input_features = input_features.to(target_device)
                input_features = input_features.to(dtype=self.audio_dtype)

                # Mask 생성 (NoneType 방지)
                if feature_attention_mask is None:
                    batch_size = input_features.shape[0]
                    time_dim = input_features.shape[2] 
                    feature_attention_mask = torch.ones(
                        (batch_size, time_dim), 
                        dtype=torch.long, 
                        device=target_device
                    )
                else:
                    if feature_attention_mask.device != target_device:
                        feature_attention_mask = feature_attention_mask.to(target_device)

                # 실제 임베딩 추출
                actual_audio_embeds = self.model.thinker.get_audio_features(
                    input_features,
                    feature_attention_mask=feature_attention_mask
                )
                
                # input_ids 생성
                actual_token_count = actual_audio_embeds.shape[1]
                audio_token_id = self.model.config.thinker_config.audio_token_id
                
                input_ids = torch.full(
                    (1, actual_token_count), 
                    audio_token_id, 
                    dtype=torch.long, 
                    device=target_device
                )
                
                inputs_embeds = actual_audio_embeds

            # 2. Text Input Case
            elif input_ids is not None:
                if input_ids.device != target_device:
                    input_ids = input_ids.to(target_device)
                # Text 모드에서는 inputs_embeds는 None 상태 유지 (모델 내부에서 생성)
                pass 
                
            else:
                raise ValueError("ThinkerStep: input_ids and input_features are both None")

            # 3. Forward
            # ★ [FIX] inputs_embeds가 None이어도 에러 안 나게 처리됨 (모델 내부 로직)
            outputs = self.model.thinker(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds, 
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            
            consumed_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
            return outputs, consumed_len

        except Exception as e:
            log("error", f"🚨 Error in thinker_step: {e}")
            import traceback
            traceback.print_exc()
            raise e # 에러를 상위로 던져서 멈추게 함 (디버깅용)

    # @torch.no_grad()
    # def talker_step(self, thinker_hidden, past_key_values, step_idx, input_ids=None):
    #     # log("debug", f"🔍 [Check] thinker_hidden: shape={thinker_hidden.shape}, device={thinker_hidden.device}, dtype={thinker_hidden.dtype}")
        
        
    #     # if torch.isnan(thinker_hidden).any():
    #     #     log("error", "💀 Critical: Thinker hidden state contains NaN!")
    #     #     raise ValueError("Thinker hidden state is NaN")

    #     # if thinker_hidden.device != self.talker_device:
    #     #     thinker_hidden = thinker_hidden.to(self.talker_device)
        
    #     # if input_ids is None:
    #     #      input_ids = torch.tensor([[self.model.config.talker_config.codec_bos_id]], device=self.talker_device)
    #     # else:
    #     #      input_ids = input_ids.to(self.talker_device)

    #     # 1. 이전 단계(Thinker)의 연산이 진짜 끝났는지, GPU가 살아있는지 확인


    #     log("debug", "1️⃣ Waiting for previous CUDA operations to finish (Synchronize)...")
    #     try:
    #         torch.cuda.synchronize() # 여기서 멈추면 -> 범인은 Thinker_step 입니다.
    #         log("debug", "✅ GPU is alive and synced.")
    #     except Exception as e:
    #         log("error", f"❌ GPU died BEFORE projection: {e}")
    #         raise e

    #     # 2. 데이터 유효성 검사 (NaN/Inf)
    #     log("debug", "2️⃣ Checking for NaN/Inf in thinker_hidden...")
    #     if torch.isnan(thinker_hidden).any() or torch.isinf(thinker_hidden).any():
    #         log("error", "💀 Critical: thinker_hidden contains NaN or Inf!")
    #         # NaN이 있으면 다음 연산이 멈출 수 있음
    #         log("debug", f"Sample values: {thinker_hidden[0,0,:10]}")
    #         raise ValueError("NaN detected in thinker output")

    #     # 3. 레이어 자체 테스트 (더미 데이터)
    #     # 만약 thinker_hidden에 뭔가 문제가 있다면 더미는 통과하고 실데이터는 멈출 것임
    #     log("debug", "3️⃣ Running Dummy Projection Test...")
    #     try:
    #         dummy_input = torch.randn_like(thinker_hidden)
    #         conditioned_hidden = self.model.talker.text_projection(dummy_input)
    #         log("debug", "✅ Dummy Projection passed (Layer is fine).")
    #     except Exception as e:
    #         log("error", f"❌ Layer weights might be corrupted: {e}")
    #         raise e

    #     # =====================================================================
    #     # 🚀 실제 로직 실행
    #     # =====================================================================
        
    #     # 1. Device Move
    #     if thinker_hidden.device != self.talker_device:
    #         thinker_hidden = thinker_hidden.to(self.talker_device)
        
    #     if input_ids is None:
    #          input_ids = torch.tensor([[self.model.config.talker_config.codec_bos_id]], device=self.talker_device)
    #     else:
    #          input_ids = input_ids.to(self.talker_device)

    #     2. Real Projection (여기서 멈추면 데이터 특이성 문제)
       
    #     conditioned_hidden = self.model.talker.text_projection(thinker_hidden)
    #     log("debug","finishing projection ") 

    #     ##finshing debugging
    #     audio_embed = self.model.talker.model.get_input_embeddings()(input_ids)
    #     talker_inputs_embeds = audio_embed + conditioned_hidden
        
    #     position_ids = torch.tensor([[step_idx]], device=self.talker_device)
    #     position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)


    #     log("debug", f"Talker Step {step_idx}: Input Embeds Shape: {talker_inputs_embeds.shape}")
    #     talker_out = self.model.talker.model(
    #         inputs_embeds=talker_inputs_embeds,
    #         past_key_values=past_key_values,
    #         position_ids=position_ids,
    #         use_cache=True
    #     )
        
    #     logits = self.model.talker.codec_head(talker_out.last_hidden_state[:, -1, :])
    #     layer0_code = logits.argmax(dim=-1, keepdim=True)
        
    #     last_id_hidden = self.model.talker.get_input_embeddings()(layer0_code)
    #     past_hidden = talker_out.last_hidden_state[:, -1:]
    #     predictor_input = torch.cat((past_hidden, last_id_hidden), dim=1)
        
    #     needed_tokens = self.num_quantizers - 1
        
    #     predictor_out = self.model.talker.code_predictor.generate(
    #         inputs_embeds=predictor_input,
    #         max_new_tokens=needed_tokens, 
    #         do_sample=False
    #     )
        
    #     full_audio_codes = torch.cat([layer0_code, predictor_out], dim=1)
    #     return full_audio_codes, talker_out.past_key_values
    @torch.no_grad()
    def talker_step(self, thinker_hidden, past_key_values, step_idx, input_ids=None):
        try:
            target_device = self.talker_device
            
            # 1. Device & Memory Safety Check
            if thinker_hidden.device != target_device:
                thinker_hidden = thinker_hidden.to(target_device)
            if not thinker_hidden.is_contiguous():
                thinker_hidden = thinker_hidden.contiguous()

            # 2. Projection (가장 많이 멈추는 구간)
            #    여기서 멈추지 않게 타임아웃을 걸 수는 없으니, 에러가 나면 더미로 대체하는 구조는 아니지만,
            #    최소한 확실하게 실행되도록 구성
            conditioned_hidden = self.model.talker.text_projection(thinker_hidden)
            
            # 3. Main Forward
            if input_ids is None:
                 input_ids = torch.tensor([[self.model.config.talker_config.codec_bos_id]], device=self.talker_device)
            else:
                 input_ids = input_ids.to(self.talker_device)

            audio_embed = self.model.talker.model.get_input_embeddings()(input_ids)
            talker_inputs_embeds = audio_embed + conditioned_hidden
            
            max_pos_limit = getattr(self.model.config.talker_config.text_config, "max_position_embeddings", 2048)
            
            # Talker는 오디오 인코더 제약이 없어서 좀 더 길 수 있지만, 
            # 안전하게 1500~2000 사이 적당한 값으로 순환 (Thinker와 비슷하게 맞추는 게 좋음)
            if max_pos_limit > 1500: max_pos_limit = 1500 # 보수적 설정
            
            cycle_start = max_pos_limit // 2
            cycle_len = max_pos_limit - cycle_start
            
            if step_idx >= max_pos_limit:
                safe_step_idx = cycle_start + (step_idx - cycle_start) % cycle_len
            else:
                safe_step_idx = step_idx
                
            if safe_step_idx >= max_pos_limit:
                safe_step_idx = max_pos_limit - 1
                
            position_ids = torch.tensor([[safe_step_idx]], device=target_device)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

            talker_out = self.model.talker.model(
                inputs_embeds=talker_inputs_embeds,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True
            )

            # 4. Code Prediction (Manual Loop)
            #    여기가 너무 느리거나 멈추면 바로 Skip하기 위해 try-except 블록 강화
            logits = self.model.talker.codec_head(talker_out.last_hidden_state[:, -1, :])
            layer0_code = logits.argmax(dim=-1, keepdim=True)
            
            # --- [BYPASS START] 복잡한 Predictor 로직 대신 단순화 ---
            # 만약 여기서도 멈춘다면 아래 전체 루프를 주석 처리하고 
            # full_audio_codes = torch.randint(0, 1024, (1, 8), device=self.talker_device) 로 대체 가능
            
            last_id_hidden = self.model.talker.get_input_embeddings()(layer0_code)
            past_hidden = talker_out.last_hidden_state[:, -1:]
            predictor_input = torch.cat((past_hidden, last_id_hidden), dim=1)
            
            predictor_codes = [layer0_code]
            predictor_kv = None 
            
            for i in range(self.num_quantizers - 1):
                # Predictor Forward
                pred_out = self.model.talker.code_predictor.model(
                    inputs_embeds=predictor_input,
                    past_key_values=predictor_kv,
                    use_cache=True
                )
                predictor_kv = pred_out.past_key_values
                
                # ★★★ [수정된 부분] 주소 변경: talker.lm_head -> talker.code_predictor.lm_head ★★★
                # Qwen3 구조상 Residual Layer 예측 헤드는 code_predictor 안에 있습니다.
                curr_logits = self.model.talker.code_predictor.lm_head[i](pred_out.last_hidden_state[:, -1, :])
                
                next_code = curr_logits.argmax(dim=-1, keepdim=True)
                predictor_codes.append(next_code)
                
                # 다음 입력 임베딩
                predictor_input = self.model.talker.code_predictor.get_input_embeddings()[i](next_code)
            
            full_audio_codes = torch.cat(predictor_codes, dim=1)
            return full_audio_codes, talker_out.past_key_values

        except Exception as e:
            # 🚨 [EMERGENCY BYPASS] 무슨 에러가 나든 멈추지 않게 가짜 데이터 리턴
            log("error", f"🚨 Talker Crashed! Returning Dummy Data. Error: {e}")
            
            # 더미 오디오 코드 (랜덤)
            dummy_codes = torch.randint(0, 1024, (1, self.num_quantizers), device=self.talker_device)
            # 더미 KV Cache (그냥 None 주면 다음 스텝에서 에러날 수 있으니, 이전꺼 리턴하거나 None)
            # 여기서는 안전하게 None 리턴 (모델이 알아서 처리하게 둠, 성능은 망가지지만 루프는 돈다)
            return dummy_codes, past_key_values

    @torch.no_grad()
    def decode_audio(self, audio_codes: torch.Tensor) -> np.ndarray:
        # [Device Alignment] Code2Wav가 있는 곳으로 이동
        target_device = self.code2wav_device
        
        if audio_codes.device != target_device:
            audio_codes = audio_codes.to(target_device)
            
        if audio_codes.dim() == 2:
            audio_codes = audio_codes.unsqueeze(-1)
            
        wav_tensor = self.model.code2wav(audio_codes)
        wav_cpu = wav_tensor.to("cpu", non_blocking=True).float().numpy()
        return wav_cpu

# =============================================================================
# 3. 엔진 클래스 (Asyncio + Executor)
# =============================================================================
class Qwen3OmniFullDuplexEngine:
    def __init__(self, model, tokenizer, config: EngineConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = config
        self.logic = Qwen3DuplexLogic(model)
        
        self.input_queue = None
        self.hidden_queue = None
        self.output_queue = None
        
        self.thinker_kv_cache = None
        self.talker_kv_cache = None
        self.last_talker_token = None
        
        self.thinker_step_count = 0
        self.talker_step_count = 0
        
        self.is_running = False
        self.thinker_task = None
        self.talker_task = None

    async def initialize(self):
        log("info", "Initializing Async Engine...")
        self.input_queue = asyncio.Queue()
        self.hidden_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()

        # System Prompt Tokenize
        initial_ids = self.tokenizer(
            self.cfg.system_prompt_text, 
            return_tensors="pt", 
            add_special_tokens=False
        ).input_ids.to(self.logic.thinker_device)
        
        # Talker 초기 토큰 설정
        codec_bos = self.model.config.talker_config.codec_bos_id
        self.last_talker_token = torch.tensor([[codec_bos]], device=self.logic.talker_device)

        # Prefill (Blocking OK here)
        with torch.no_grad():
            # ★ [수정 포인트] 리턴값이 (outputs, seq_len) 튜플이므로 언패킹 해야 함
            out, _ = self.logic.thinker_step(
                input_ids=initial_ids, 
                input_features=None, 
                feature_attention_mask=None,
                past_key_values=None, 
                step_idx=0
            )
            
            # 이제 out은 정상적인 ModelOutput 객체입니다.
            self.thinker_kv_cache = out.past_key_values
            self.thinker_step_count = initial_ids.shape[1]
            
        log("info", "Engine Ready.")
        
    async def _thinker_loop(self):
        loop = asyncio.get_running_loop()
        
        while self.is_running:
            audio_features = await self.input_queue.get()
            
            def run_thinker_inference():
                with torch.no_grad():
                    # =========================================================
                    # [Step 1] 듣기 (Listening)
                    # =========================================================
                    thinker_out, consumed_len = self.logic.thinker_step(
                        input_ids=None, 
                        input_features=audio_features,
                        feature_attention_mask=None,
                        past_key_values=self.thinker_kv_cache,
                        step_idx=self.thinker_step_count
                    )
                    
                    self.thinker_kv_cache = thinker_out.past_key_values
                    self.thinker_step_count += consumed_len 

                    # =========================================================
                    # [Step 2] 판단 (Decision)
                    # =========================================================
                    next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    token_id = next_token.item()

                    log("debug", f"Thinker predicted token ID: {token_id}")
                    # if token_id == self.cfg.silence_token_id or token_id == 151645:
                    #      return None, "<|silence|>"

                    # =========================================================
                    # [Step 3] 말하기 (Speaking) - ★ 수정된 부분
                    # =========================================================
                    current_turn_hiddens = []
                    
                    # [삭제됨] current_turn_hiddens.append(thinker_out.hidden_states[-1]) 
                    # 이유: 위 코드는 '오디오'에 대한 히든 스테이트이므로 Talker에게 적합하지 않음.
                    
                    # [수정] 설정된 토큰 수(예: 2)만큼 루프를 돌며 "순수 텍스트 히든 스테이트" 생성
                    token_str = ""
                    
                    # range(N - 1) -> range(N)으로 변경
                    for _ in range(self.cfg.text_output_tokens):
                        # 1. 예측된 텍스트 토큰(next_token)을 입력으로 다시 Thinker 실행
                        thinker_out, _ = self.logic.thinker_step(
                            input_ids=next_token,
                            input_features=None,
                            feature_attention_mask=None,
                            past_key_values=self.thinker_kv_cache,
                            step_idx=self.thinker_step_count
                        )
                        
                        # 2. 상태 업데이트
                        self.thinker_kv_cache = thinker_out.past_key_values
                        self.thinker_step_count += 1
                        
                        # 3. ★ 중요: 텍스트 입력에 대한 결과(Hidden State)만 저장
                        #    이것이 공식 코드의 assistant_hidden 부분과 일치함
                        safe_hidden = thinker_out.hidden_states[-1].detach().clone()


                        current_turn_hiddens.append(safe_hidden)
                        
                        # 4. 다음 토큰 예측 및 로그 준비
                        next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        token_str += self.tokenizer.decode([next_token.item()])
                    
                    final_hidden_to_send = torch.cat(current_turn_hiddens, dim=1).contiguous()
                    # 결과: [1, 2, Hidden_Dim] (오디오 섞이지 않은 순수 텍스트 상태)
                    return final_hidden_to_send, token_str

            # Executor 실행 (Sender를 방해하지 않음)
            stacked_hidden, log_str = await loop.run_in_executor(None, run_thinker_inference)
            
            # 실시간 토큰 로그 출력
            get_logger().print_token(log_str)

            # ★ [결과 처리] Hidden State가 있을 때만(Silence가 아닐 때만) 큐에 넣음
            if stacked_hidden is not None:
                await self.hidden_queue.put(stacked_hidden)
            else:
                # Silence인 경우: 큐에 넣지 않고 루프 처음으로 돌아감 (다음 오디오 대기)
                # 하지만 KV Cache는 이미 업데이트 되었으므로 문맥은 이어짐
                pass

    async def _talker_loop(self):
        log("info", "Talker Loop Started")
        loop = asyncio.get_running_loop()
        
        while self.is_running:
            # 큐에서 데이터를 꺼낼 때까지 대기
            source_hidden = await self.hidden_queue.get()
            
            def run_talker_inference():
                with torch.no_grad():
                    num_hiddens = source_hidden.shape[1]
                    ratio = self.cfg.audio_output_tokens // self.cfg.text_output_tokens
                    output_chunks = []

            

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
                            self.last_talker_token = codes[:, 0:1] 
                            
                            wav_np = self.logic.decode_audio(codes)
                            output_chunks.append(wav_np)
                    return output_chunks

            # GPU 연산 수행
            wav_chunks_np = await loop.run_in_executor(None, run_talker_inference)
            
            # 결과 전송
            for wav_np in wav_chunks_np:
                wav_int16 = (wav_np * 32767).astype(np.int16).tobytes()
                await self.output_queue.put(wav_int16)

    async def start(self):
        if self.is_running: return
        self.is_running = True
        await self.initialize()
        self.thinker_task = asyncio.create_task(self._thinker_loop())
        self.talker_task = asyncio.create_task(self._talker_loop())
        log("info", "Engine Started (Async + Executor)")

    async def stop(self):
        self.is_running = False
        if self.thinker_task: self.thinker_task.cancel()
        if self.talker_task: self.talker_task.cancel()
        log("info", "Engine Stopped")

    async def push_audio(self, audio_features: torch.Tensor):
        await self.input_queue.put(audio_features)

    async def get_audio_output(self) -> Optional[bytes]:
        try:
            return self.output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
