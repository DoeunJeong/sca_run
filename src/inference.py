import torch
import numpy as np
import asyncio
from dataclasses import dataclass
from typing import Optional

# Moshi 스타일 로거
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
    # ★ [오디오 고정 길이 설정] 0.32초 = 4토큰 (Qwen3-Omni 구조적 특성)
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
        self.device = model.device 
        
        # [필수] Attention 충돌(RuntimeError) 방지를 위한 Eager 모드 강제 설정
        # 모델 내부 로직을 타더라도 이 설정은 꼭 필요합니다.
        def force_eager(module):
            if hasattr(module, "config"):
                module.config._attn_implementation = "eager"
        
        force_eager(self.model)
        if hasattr(self.model, "thinker"): force_eager(self.model.thinker)
        if hasattr(self.model, "talker"): force_eager(self.model.talker.model)

        # Device Mapping
        if hasattr(model, "thinker"):
            self.thinker_device = model.thinker.device
        else:
            self.thinker_device = self.device

        if hasattr(model, "talker"):
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

    # _calc_audio_token_count 함수 삭제됨 (요청 반영)

    @torch.no_grad()
    def thinker_step(self, input_ids, input_features, feature_attention_mask, past_key_values, fixed_audio_tokens=4):
        """
        Thinker Step: 
        - 오디오/텍스트 입력을 받아 모델 내부 로직(Forward)을 통해 다음 토큰 예측
        - 오디오 길이는 4토큰(0.32초)으로 고정 가정
        """
        target_device = self.thinker_device
        
        try:
            # -----------------------------------------------------------------
            # Case 1: Audio Input Processing
            # -----------------------------------------------------------------
            if input_features is not None:
                if input_features.device != target_device:
                    input_features = input_features.to(target_device)
                input_features = input_features.to(dtype=self.audio_dtype)

                # [Mask 자동 생성] NoneType 에러 방지
                if feature_attention_mask is None:
                    # shape: [Batch, Mel, Time]
                    feature_attention_mask = torch.ones(
                        (input_features.shape[0], input_features.shape[2]), 
                        dtype=torch.long, 
                        device=target_device
                    )
                else:
                    if feature_attention_mask.device != target_device:
                        feature_attention_mask = feature_attention_mask.to(target_device)

                # [Input IDs 생성] "오디오는 항상 4토큰" (Config 사용)
                audio_token_id = self.model.config.thinker_config.audio_token_id
                
                # [Batch=1, Length=4]
                input_ids = torch.full(
                    (1, fixed_audio_tokens), 
                    audio_token_id, 
                    dtype=torch.long, 
                    device=target_device
                )
                
                # ★ 핵심: inputs_embeds를 직접 계산하지 않고 None으로 둠.
                # 대신 input_features를 forward에 전달하여 모델이 내부적으로 처리하게 함.
                inputs_embeds = None 

            # -----------------------------------------------------------------
            # Case 2: Text Input Processing
            # -----------------------------------------------------------------
            elif input_ids is not None:
                if input_ids.device != target_device:
                    input_ids = input_ids.to(target_device)
                
                # 텍스트는 Features 없음
                input_features = None
                inputs_embeds = None
                
            else:
                raise ValueError("ThinkerStep: Both input_ids and input_features are None")

            # -----------------------------------------------------------------
            # Forward (Model Internal Logic)
            # -----------------------------------------------------------------
            # position_ids와 rope_deltas를 전달하지 않음 -> 모델이 내부에서 past_key_values 길이를 보고 자동 계산
            # input_features를 전달함 -> 모델이 내부에서 get_audio_features -> Projection 수행 (차원 불일치 해결)
            
            outputs = self.model.thinker(
                input_ids=input_ids,
                input_features=input_features,       # 오디오 원본 전달 (내부 처리 유도)
                feature_attention_mask=feature_attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,         # None (모델이 계산)
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            
            return outputs

        except Exception as e:
            log("error", f"🚨 Error in thinker_step: {e}")
            import traceback
            traceback.print_exc()
            raise e

    @torch.no_grad()
    def talker_step(self, thinker_hidden, past_key_values, input_ids=None):
        """
        Talker Step:
        - step_idx 등 위치 관련 변수 제거 (모델 내부 및 KV Cache에 위임)
        """
        try:
            target_device = self.talker_device
            
            if thinker_hidden.device != target_device:
                thinker_hidden = thinker_hidden.to(target_device)
            if not thinker_hidden.is_contiguous():
                thinker_hidden = thinker_hidden.contiguous()

            # Projection
            conditioned_hidden = self.model.talker.text_projection(thinker_hidden)
            
            if input_ids is None:
                 input_ids = torch.tensor([[self.model.config.talker_config.codec_bos_id]], device=target_device)
            else:
                 if input_ids.device != target_device:
                     input_ids = input_ids.to(target_device)

            # Embedding Sum
            audio_embed = self.model.talker.model.get_input_embeddings()(input_ids)
            talker_inputs_embeds = audio_embed + conditioned_hidden
            
            # Forward (position_ids 없이 호출 -> 자동 관리)
            talker_out = self.model.talker.model(
                inputs_embeds=talker_inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )

            # Code Prediction (RVQ)
            logits = self.model.talker.codec_head(talker_out.last_hidden_state[:, -1, :])
            layer0_code = logits.argmax(dim=-1, keepdim=True)
            
            last_id_hidden = self.model.talker.get_input_embeddings()(layer0_code)
            past_hidden = talker_out.last_hidden_state[:, -1:]
            predictor_input = torch.cat((past_hidden, last_id_hidden), dim=1)
            
            predictor_codes = [layer0_code]
            predictor_kv = None 
            
            for i in range(self.num_quantizers - 1):
                pred_out = self.model.talker.code_predictor.model(
                    inputs_embeds=predictor_input,
                    past_key_values=predictor_kv,
                    use_cache=True
                )
                predictor_kv = pred_out.past_key_values
                
                curr_logits = self.model.talker.code_predictor.lm_head[i](pred_out.last_hidden_state[:, -1, :])
                next_code = curr_logits.argmax(dim=-1, keepdim=True)
                predictor_codes.append(next_code)
                
                predictor_input = self.model.talker.code_predictor.get_input_embeddings()[i](next_code)
            
            full_audio_codes = torch.cat(predictor_codes, dim=1)
            return full_audio_codes, talker_out.past_key_values

        except Exception as e:
            log("error", f"🚨 Talker Crashed! {e}")
            dummy_codes = torch.randint(0, 1024, (1, self.num_quantizers), device=self.talker_device)
            return dummy_codes, past_key_values

    @torch.no_grad()
    def decode_audio(self, audio_codes: torch.Tensor) -> np.ndarray:
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
        
        self.input_queue = asyncio.Queue()
        self.hidden_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        
        # ★ [간소화] step_count 관리 제거 (모델 KV Cache가 알아서 함)
        self.thinker_kv_cache = None
        self.talker_kv_cache = None
        self.last_talker_token = None
        
        self.is_running = False
        self.thinker_task = None
        self.talker_task = None

    async def initialize(self):
        log("info", "Initializing Async Engine...")
        self.input_queue = asyncio.Queue()
        self.hidden_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()

        initial_ids = self.tokenizer(
            self.cfg.system_prompt_text, 
            return_tensors="pt", 
            add_special_tokens=False
        ).input_ids.to(self.logic.thinker_device)
        
        codec_bos = self.model.config.talker_config.codec_bos_id
        self.last_talker_token = torch.tensor([[codec_bos]], device=self.logic.talker_device)

        # Prefill
        with torch.no_grad():
            # step_idx 제거
            out = self.logic.thinker_step(
                input_ids=initial_ids, 
                input_features=None, 
                feature_attention_mask=None,
                past_key_values=None
            )
            self.thinker_kv_cache = out.past_key_values
            
        log("info", "Engine Ready.")
        
    async def _thinker_loop(self):
        log("info", "🚀 Thinker Loop Started")
        loop = asyncio.get_running_loop()
        
        while self.is_running:
            audio_features = await self.input_queue.get()
            
            def run_thinker_inference():
                try:
                    with torch.no_grad():
                        # [Step 1] 듣기 (Listening)
                        # 오디오는 항상 고정 길이(Config 참조)라고 가정하고 넘김
                        thinker_out = self.logic.thinker_step(
                            input_ids=None, 
                            input_features=audio_features,
                            feature_attention_mask=None,
                            past_key_values=self.thinker_kv_cache,
                            fixed_audio_tokens=self.cfg.audio_input_tokens
                        )
                        self.thinker_kv_cache = thinker_out.past_key_values

                        # [Step 2] 판단 (Decision)
                        next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        token_id = next_token.item()
                        log("debug", f"🧠 Thinker predicted: {token_id}")

                        # [Step 3] 말하기 (Speaking)
                        current_turn_hiddens = []
                        token_str = ""
                        
                        for _ in range(self.cfg.text_output_tokens):
                            # Text Generation
                            thinker_out = self.logic.thinker_step(
                                input_ids=next_token,
                                input_features=None,
                                feature_attention_mask=None,
                                past_key_values=self.thinker_kv_cache
                            )
                            self.thinker_kv_cache = thinker_out.past_key_values
                            
                            safe_hidden = thinker_out.hidden_states[-1].detach().clone()
                            current_turn_hiddens.append(safe_hidden)
                            
                            next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                            token_str += self.tokenizer.decode([next_token.item()])
                        
                        final_hidden_to_send = torch.cat(current_turn_hiddens, dim=1).contiguous()
                        return final_hidden_to_send, token_str
                except Exception as e:
                    log("error", f"💥 Thinker Loop Crashed: {e}")
                    import traceback
                    traceback.print_exc()
                    return None, ""

            stacked_hidden, log_str = await loop.run_in_executor(None, run_thinker_inference)
            
            if log_str:
                get_logger().print_token(log_str)

            if stacked_hidden is not None:
                await self.hidden_queue.put(stacked_hidden)

    async def _talker_loop(self):
        log("info", "Talker Loop Started")
        loop = asyncio.get_running_loop()
        
        while self.is_running:
            source_hidden = await self.hidden_queue.get()
            
            def run_talker_inference():
                with torch.no_grad():
                    num_hiddens = source_hidden.shape[1]
                    ratio = self.cfg.audio_output_tokens // self.cfg.text_output_tokens
                    output_chunks = []

                    for i in range(num_hiddens):
                        one_hidden = source_hidden[:, i:i+1, :]
                        for _ in range(ratio):
                            # Talker Step (step_idx 없음)
                            codes, new_kv = self.logic.talker_step(
                                thinker_hidden=one_hidden,
                                past_key_values=self.talker_kv_cache,
                                input_ids=self.last_talker_token
                            )
                            self.talker_kv_cache = new_kv
                            self.last_talker_token = codes[:, 0:1] 
                            
                            wav_np = self.logic.decode_audio(codes)
                            output_chunks.append(wav_np)
                    return output_chunks

            wav_chunks_np = await loop.run_in_executor(None, run_talker_inference)
            
            for wav_np in wav_chunks_np:
                wav_int16 = (wav_np * 32767).astype(np.int16).tobytes()
                await self.output_queue.put(wav_int16)

    async def start(self):
        if self.is_running: return
        self.is_running = True
        await self.initialize()
        self.thinker_task = asyncio.create_task(self._thinker_loop())
        self.talker_task = asyncio.create_task(self._talker_loop())
        log("info", "Engine Started (Clean Full-Duplex)")

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