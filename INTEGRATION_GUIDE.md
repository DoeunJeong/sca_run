# Qwen3-Omni FullDuplex 통합 가이드

## 📋 변경 사항 요약

### 1. `sca_run/team_infer.py` 전체 수정
**기존:** 비어있던 함수
**변경:** 팀원의 Qwen3DuplexLogic과 통합

**핵심 기능:**
- Qwen3-Omni 모델 로드
- Thinker(오디오 이해) 실행
- Talker(답변 생성) 실행  
- Code2Wav(음성 생성) 실행
- Log-Mel Spectrogram → 음성 변환

### 2. 전역 상태 관리 추가
```python
_model_lock = threading.Lock()          # 멀티스레드 안전성
_qwen_model = None                      # 로드된 모델
_duplex_logic = None                    # Qwen3DuplexLogic 인스턴스
_step_count = 0                         # 추론 단계 카운터
```

---

## 🔧 설정 방법

### 방법 1: 환경 변수 설정 (권장)

**PowerShell에서:**
```powershell
# 파인튜닝된 모델 경로
$env:SCA_QWEN_MODEL_ID = "path/to/your/finetuned/model"

# 또는 HuggingFace 모델
$env:SCA_QWEN_MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# GPU 설정
$env:SCA_QWEN_DEVICE_MAP = "cuda:0"

# 데이터 타입 (메모리 절약)
$env:SCA_QWEN_TORCH_DTYPE = "float16"

# 서버 실행
python -m sca_run.server --config config/default.toml
```

**Linux/Mac에서:**
```bash
export SCA_QWEN_MODEL_ID="path/to/your/finetuned/model"
export SCA_QWEN_DEVICE_MAP="auto"
export SCA_QWEN_TORCH_DTYPE="float16"

python -m sca_run.server --config config/default.toml
```

### 방법 2: `config/default.toml` 수정

```toml
[qwen]
backend = "team"
model_id = "path/to/your/finetuned/model"
device_map = "cuda:0"
torch_dtype = "float16"
```

---

## 📊 데이터 흐름

```
웹 UI (index.html)
   ↓ PCM16 음성 (16kHz)
   
server.py (/ws/pcm16)
   ↓ 청크된 PCM16
   
audio_chunker.py
   ↓ 고정 크기 PCM16 (320ms)
   
feature_extractor.py
   ↓ Log-Mel Spectrogram [1, 128, T]
   
qwen_client.py
   ↓ AudioInput(features)
   
team_infer.py (✨ 새로 구현)
   ├─ _load_qwen_model()
   │  └─ 팀원의 파인튜닝 모델 로드
   │
   ├─ _init_duplex_logic()
   │  └─ Qwen3DuplexLogic 초기화
   │
   └─ infer_team_wav()
      ├─ thinker_step() → 오디오 이해 [1, 128, T] → 텍스트 토큰
      ├─ talker_step()  → 텍스트 → 오디오 코드
      ├─ code2wav()     → 오디오 코드 → 음성 [T]
      └─ float32로 변환
      
qwen_client.py
   ↓ TeamAudioReturn(wav float32, sr=24000)
   
server.py
   ↓ PCM16LE로 변환
   
index.html (✨ 스피커)
   └─ 실시간 재생!
```

---

## 🚀 실행 방법

### 1단계: 통합 확인
```bash
python setup_integration.py
```

이 스크립트가 확인하는 것:
- ✅ 팀원의 코드 파일 존재 여부
- ✅ 필수 패키지 설치 여부
- ✅ 필요한 환경 변수 설정

### 2단계: 모델 로드 및 서버 시작
```bash
python -m sca_run.server --config config/default.toml --host 0.0.0.0 --port 8000
```

### 3단계: 웹 브라우저에서 접속
```
http://localhost:8000
```

### 4단계: 마이크로 말하면 AI가 답변!

---

## 📊 함수 구조

### `team_infer.py` 내부 함수들

#### 1. `_load_qwen_model(cfg: AppConfig) -> torch.nn.Module`
**역할:** Qwen3-Omni 모델 로드
**입력:** AppConfig
**출력:** 로드된 모델
**처리:**
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
)
```

#### 2. `_init_duplex_logic(cfg: AppConfig) -> Qwen3DuplexLogic`
**역할:** Qwen3DuplexLogic 초기화 (한 번만)
**입력:** AppConfig
**출력:** 팀원의 Qwen3DuplexLogic 인스턴스
**처리:**
```python
model = _load_qwen_model(cfg)  # 모델 로드
logic = Qwen3DuplexLogic(model)  # 팀원의 클래스 사용
```

#### 3. `infer_team_wav(cfg: AppConfig, audio_in: AudioInput) -> Optional[TeamAudioReturn]`
**역할:** 실제 추론 수행 (메인 함수)
**입력:** 
- `cfg`: 설정
- `audio_in`: AudioInput(features=[1,128,T])

**출력:** TeamAudioReturn(wav float32, sr=24000)

**처리 순서:**
```
1. Duplex Logic 초기화
2. Feature 준비 (CPU/GPU 체크)
3. Thinker 실행: 오디오 이해
4. Talker 실행: 답변 생성
5. Code2Wav 실행: 음성 생성
6. 반환
```

#### 4. `reset_conversation()`
**역할:** 대화 상태 초기화
**사용:** 새로운 대화 시작 시

---

## ⚠️ 주의사항

### 1. 첫 번째 호출이 느린 이유
```
첫 번째 호출:
  모델 로드 (5~30초) → Duplex Logic 초기화 → 추론
  
이후 호출:
  추론만 (1~2초)
```
✅ 이것은 정상입니다. 대화를 시작하면 점점 빨라집니다.

### 2. GPU 메모리 부족 시
```powershell
# 데이터 타입을 float16으로 변경
$env:SCA_QWEN_TORCH_DTYPE = "float16"

# 또는 8-bit 양자화
$env:SCA_QWEN_LOAD_IN_8BIT = "true"
```

### 3. 모델이 로드되지 않는 경우
```bash
# 모델 다운로드 테스트
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-Omni-30B-A3B-Instruct', trust_remote_code=True)"
```

---

## 🔍 디버깅

### 로그 확인
```bash
# 전체 로그 확인
python -m sca_run.server 2>&1 | Tee-Object -FilePath debug.log

# 또는 linux
python -m sca_run.server 2>&1 | tee debug.log
```

### Team Inference 로그 메시지
```
[Team Inference] 🔄 Qwen3-Omni 모델 로딩 중...
[Team Inference] Model ID: ...
[Team Inference] ✅ 모델 로드 완료!
[Team Inference] ✅ Qwen3DuplexLogic 초기화 완료!
[Team Inference] 입력 Feature 형태: torch.Size([1, 128, T])
[Team Inference] 🧠 Thinker 처리 중...
[Team Inference] 👄 Talker 처리 중...
[Team Inference] 🎵 Code2Wav 처리 중...
[Team Inference] ✅ 생성된 음성 길이: ... samples (...초)
```

---

## 📈 성능 최적화

### 1. 배치 처리
현재 코드는 한 번에 1개의 청크만 처리합니다.
```python
# 여러 청크를 모아서 처리하려면 (미구현)
features_batch = torch.cat([f1, f2, f3], dim=0)
```

### 2. 스트리밍 최적화
팀원의 `src/interface.py`의 `thinker_output_queue`를 활용:
```python
# Thinker 결과를 Queue에 넣기
state.thinker_output_queue.put(thinker_out.hidden_states[-1])

# Talker가 Queue에서 꺼내기
while not state.thinker_output_queue.empty():
    hidden = state.thinker_output_queue.popleft()
    # ... Talker 처리
```

---

## 📝 추가 설정

### `config/default.toml` 전체 설정
```toml
[audio]
sample_rate = 16000
frame_hz = 12.5
frames_per_chunk = 4
channels = 1
sample_width_bytes = 2

[qwen]
backend = "team"
model_id = "your/finetuned/model"
device_map = "cuda:0"
torch_dtype = "float16"
```

---

## 🎯 다음 단계

### 1단계: 통합 확인
```bash
python setup_integration.py
```

### 2단계: 모델 경로 설정
```bash
$env:SCA_QWEN_MODEL_ID = "your_model_path"
```

### 3단계: 서버 시작
```bash
python -m sca_run.server --config config/default.toml
```

### 4단계: 테스트
브라우저에서 http://localhost:8000 접속

---

## 💡 문제 해결

### "팀원의 inference.py를 찾을 수 없습니다"
```
해결: src/inference.py가 sca_run과 같은 폴더 구조에 있는지 확인
현재: sca_run/
      src/
          inference.py  ✅
          interface.py  ✅
```

### "CUDA out of memory"
```
해결: float16 사용
$env:SCA_QWEN_TORCH_DTYPE = "float16"
```

### "모델이 너무 느림"
```
해결: 더 작은 모델 사용
$env:SCA_QWEN_MODEL_ID = "Qwen/Qwen3-Omni-10B-Instruct"
```

---

## 📞 지원

문제가 발생하면:
1. `setup_integration.py` 실행으로 상태 확인
2. 디버그 로그 확인
3. 환경 변수 재확인

