import torch
import numpy as np
import asyncio
from dataclasses import dataclass
from typing import Optional

# Moshi 스타일 로거 임포트 (기존 유지)
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
# 2. 로직 클래스 (Stateless Tensor Operations) - [수정 완료]
# =============================================================================
class Qwen3DuplexLogic:
    def __init__(self, model):
        self.model = model
        self.device = model.device 
        
        # 분산 환경 고려한 디바이스 매핑
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

    # [삭제됨] _calc_audio_token_count 제거 (get_audio_features 결과 사용)

    @torch.no_grad()
    def thinker_step(self, input_ids, input_features, feature_attention_mask, past_key_values):
        """
        Thinker Step: Audio or Text Input -> Next Token Prediction
        [변경점] 
        1. step_idx 인자 제거: position_ids 수동 관리 안 함
        2. get_audio_features 활용: 실제 오디오 토큰 수 자동 반영
        3. inputs_embeds 직접 주입: 인덱스 에러 방지
        """
        target_device = self.thinker_device
        inputs_embeds = None
        
        # ---------------------------------------------------------------------
        # Case A: Audio Input Processing
        # ---------------------------------------------------------------------
        if input_features is not None:
            if input_features.device != target_device:
                input_features = input_features.to(target_device)
            input_features = input_features.to(dtype=self.audio_dtype)

            # Mask 처리 (Time 축)
            if feature_attention_mask is None:
                # [Batch, Mel, Time] -> Time=dim 2
                audio_seq_len = torch.tensor([input_features.shape[2]], device=target_device)
            else:
                if feature_attention_mask.device != target_device:
                    feature_attention_mask = feature_attention_mask.to(target_device)
                audio_seq_len = feature_attention_mask.sum(dim=1)

            # ★ 핵심: 실제 오디오 임베딩 추출 (길이 자동 결정)
            actual_audio_embeds = self.model.thinker.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_seq_len
            )
            
            # shape 맞추기용 input_ids 생성
            actual_token_count = actual_audio_embeds.shape[1]
            audio_token_id = self.model.config.thinker_config.audio_token_id
            
            input_ids = torch.full(
                (1, actual_token_count), 
                audio_token_id, 
                dtype=torch.long, 
                device=target_device
            )
            
            # 임베딩 교체
            inputs_embeds = actual_audio_embeds

        # ---------------------------------------------------------------------
        # Case B: Text Input Processing
        # ---------------------------------------------------------------------
        elif input_ids is not None:
            if input_ids.device != target_device:
                input_ids = input_ids.to(target_device)
            # Text 모드: inputs_embeds=None 상태 유지 (모델 내부에서 생성)
            
        else:
            raise ValueError("ThinkerStep: input_ids and input_features are both None")

        # ---------------------------------------------------------------------
        # Forward Pass (No manual position_ids)
        # ---------------------------------------------------------------------
        # past_key_values가 있으면 모델이 알아서 position_ids를 이어 붙임
        outputs = self.model.thinker(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds, 
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True
        )
        
        return outputs

    @torch.no_grad()
    def talker_step(self, thinker_hidden, past_key_values, input_ids=None):
        """
        Talker Step: Thinker Hidden -> Audio Code Generation
        [변경점]
        1. step_idx 인자 제거
        2. position_ids, max_pos_limit 관련 로직 전면 삭제
        """
        target_device = self.talker_device
        
        # 1. Device & Memory Safety
        if thinker_hidden.device != target_device:
            thinker_hidden = thinker_hidden.to(target_device)
        if not thinker_hidden.is_contiguous():
            thinker_hidden = thinker_hidden.contiguous()

        # 2. Projection (Thinker Dim -> Talker Dim)
        conditioned_hidden = self.model.talker.text_projection(thinker_hidden)
        
        # 3. Talker Input Prep
        if input_ids is None:
             input_ids = torch.tensor([[self.model.config.talker_config.codec_bos_id]], device=target_device)
        else:
             if input_ids.device != target_device:
                 input_ids = input_ids.to(target_device)

        # 4. Embeddings Combine
        audio_embed = self.model.talker.model.get_input_embeddings()(input_ids)
        talker_inputs_embeds = audio_embed + conditioned_hidden
        
        # 5. Talker Forward (No manual position_ids)
        # Talker도 Causal LM이므로 KV Cache만 잘 주면 알아서 연산함
        talker_out = self.model.talker.model(
            inputs_embeds=talker_inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )

        # 6. Code Predictor (Residual Quantization)
        logits = self.model.talker.codec_head(talker_out.last_hidden_state[:, -1, :])
        layer0_code = logits.argmax(dim=-1, keepdim=True)
        
        last_id_hidden = self.model.talker.get_input_embeddings()(layer0_code)
        past_hidden = talker_out.last_hidden_state[:, -1:]
        predictor_input = torch.cat((past_hidden, last_id_hidden), dim=1)
        
        predictor_codes = [layer0_code]
        predictor_kv = None 
        
        # Autoregressive Loop for RVQ
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
# 3. 엔진 클래스 (Asyncio + Executor) - [수정 완료: step_count 제거]
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
        
        # ★ KV Cache만 유지 (step_count 삭제)
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

        # System Prompt
        initial_ids = self.tokenizer(
            self.cfg.system_prompt_text, 
            return_tensors="pt", 
            add_special_tokens=False
        ).input_ids.to(self.logic.thinker_device)
        
        # Talker Init
        codec_bos = self.model.config.talker_config.codec_bos_id
        self.last_talker_token = torch.tensor([[codec_bos]], device=self.logic.talker_device)

        # Prefill (step_idx 없이 호출)
        with torch.no_grad():
            out = self.logic.thinker_step(
                input_ids=initial_ids, 
                input_features=None, 
                feature_attention_mask=None,
                past_key_values=None
            )
            self.thinker_kv_cache = out.past_key_values
            
        log("info", "Engine Ready.")
        
    async def _thinker_loop(self):
        loop = asyncio.get_running_loop()
        
        while self.is_running:
            audio_features = await self.input_queue.get()
            
            def run_thinker_inference():
                with torch.no_grad():
                    # ---------------------------------------------------------
                    # [Step 1] 듣기 (Listening) - step_idx 없이 호출
                    # ---------------------------------------------------------
                    thinker_out = self.logic.thinker_step(
                        input_ids=None, 
                        input_features=audio_features,
                        feature_attention_mask=None,
                        past_key_values=self.thinker_kv_cache
                    )
                    
                    self.thinker_kv_cache = thinker_out.past_key_values

                    # ---------------------------------------------------------
                    # [Step 2] 판단 (Decision)
                    # ---------------------------------------------------------
                    next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    token_id = next_token.item()
                    
                    # [옵션] Silence 토큰 처리 (필요시 주석 해제)
                    # if token_id == self.cfg.silence_token_id:
                    #     return None, "<|silence|>"

                    log("debug", f"Thinker predicted: {token_id}")

                    # ---------------------------------------------------------
                    # [Step 3] 말하기 (Speaking) - 순수 텍스트 생성
                    # ---------------------------------------------------------
                    current_turn_hiddens = []
                    token_str = ""
                    
                    for _ in range(self.cfg.text_output_tokens):
                        # Text Generation (step_idx 없이 호출)
                        thinker_out = self.logic.thinker_step(
                            input_ids=next_token,
                            input_features=None,
                            feature_attention_mask=None,
                            past_key_values=self.thinker_kv_cache
                        )
                        
                        self.thinker_kv_cache = thinker_out.past_key_values
                        
                        # Talker에게 보낼 Hidden State 저장
                        safe_hidden = thinker_out.hidden_states[-1].detach().clone()
                        current_turn_hiddens.append(safe_hidden)
                        
                        # Next Token
                        next_token = thinker_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        token_str += self.tokenizer.decode([next_token.item()])
                    
                    final_hidden_to_send = torch.cat(current_turn_hiddens, dim=1).contiguous()
                    return final_hidden_to_send, token_str

            stacked_hidden, log_str = await loop.run_in_executor(None, run_thinker_inference)
            
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
                    # Audio/Text 비율에 따라 반복 생성 (보통 4:2 = 2배)
                    ratio = self.cfg.audio_output_tokens // self.cfg.text_output_tokens
                    output_chunks = []

                    for i in range(num_hiddens):
                        one_hidden = source_hidden[:, i:i+1, :]
                        for _ in range(ratio):
                            # Talker Step (step_idx 없이 호출)
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