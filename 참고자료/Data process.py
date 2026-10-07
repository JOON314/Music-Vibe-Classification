# %%
import os
import glob
import pandas as pd

# 1. CSV 파일들이 있는 폴더 경로 설정 (현재 폴더로 설정)
# 실제 파일들이 있는 경로로 수정해주세요. 예: folder_path = '/path/to/csv_files'
folder_path = 'dataset/features/features' 

# 해당 폴더 안의 모든 .csv 파일 목록 가져오기
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# 결과를 저장할 리스트
processed_data = []

for file in csv_files:
    # 파일 이름만 추출
    file_name = os.path.basename(file)
    # 2. 데이터 불러오기
    # (;을 구분자로 설정)
    df = pd.read_csv(file, sep=';')
    
    # 3. 'frameTime' 컬럼 삭제
    if 'frameTime' in df.columns:
        df = df.drop(columns=['frameTime'])
        
    # 4. 나머지 컬럼들의 평균 계산 (숫자형 데이터만)
    mean_values = df.mean(numeric_only=True)
    
    # 5. 평균값 시리즈를 1줄짜리 데이터프레임으로 변환
    mean_df = pd.DataFrame([mean_values])
    
    # 6. 맨 앞에 파일 이름을 나타내는 컬럼 추가
    mean_df.insert(0, 'FileName', file_name)
    
    # 리스트에 추가
    processed_data.append(mean_df)

# 7. 리스트에 모인 모든 데이터프레임을 하나로 합치기
if processed_data:
    final_df = pd.concat(processed_data, ignore_index=True)
    
    # 8. 최종 결과를 하나의 CSV 파일로 저장
    output_filename = 'combined_mean_results.csv'
    final_df.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"성공적으로 {len(csv_files)}개의 파일을 처리하여 '{output_filename}'로 저장했습니다.")
    
    # 최종 데이터프레임 구조 확인
    print(final_df.head())
else:
    print("처리할 CSV 파일을 찾을 수 없습니다.")

# %%
df = pd.read_csv('combined_mean_results.csv')
df.iloc[:, 0] = df.iloc[:, 0].str[:-4].astype(int)  # 뒤 3글자 제거 후 숫자 변환
df = df.sort_values(by=df.columns[0])       # 정렬
df.to_csv("combined_mean_results.csv", index=False)         # 다시 저장

# %%



