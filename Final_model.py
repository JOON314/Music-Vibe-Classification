# %%
import os
import sys
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


PYTHON = sys.executable

STEPS = [

    ("1단계: 가사 크롤링",  "model2/crawling.py"),
    ("2단계: Feature 추출", "model2/Music_analyze.py"),
    ("3단계: 딥러닝 학습 + 예측",  "model2/Deep_learning.py"),
    ("4단계: 키워드 추출",  "model2/core_word.py"),
    ("5단계: 키워드 분석",  "model2/emotion.py"),
    ("6단계: 분석결과 정리 및 시각화",  "model2/MC.py")
    
    
]

def run(label, script):
    path = os.path.join(BASE_DIR, script)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 스크립트 없음: {path}")

    print("\n" + "=" * 70)
    print(f"▶ {label}")
    print(f"  파일: {script}")
    print(f"  시작: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)

    result = subprocess.run([PYTHON, path], cwd=BASE_DIR)

    if result.returncode != 0:
        print(f"\n❌ '{script}' 실패 (exit code {result.returncode}). 파이프라인 중단.")
        sys.exit(result.returncode)

    print(f"✅ '{script}' 완료")

def main():
    t0 = datetime.now()
    print(f"🚀 전체 파이프라인 시작: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    for label, script in STEPS:
        run(label, script)
    elapsed = datetime.now() - t0
    print(f"\n🎉 전체 완료 (총 소요 {elapsed})")

if __name__ == "__main__":
    main()



