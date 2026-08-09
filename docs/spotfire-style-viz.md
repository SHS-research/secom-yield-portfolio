# Spotfire-style 시각화 (Python 재현)

> 양산 현업에서 쓰는 **TIBCO Spotfire의 대표 차트 유형**(Box Plot · Scatter Plot Matrix ·
> Treemap · 파라미터 매핑 산점도)을 SECOM 실데이터로 재현한 결과.
> 생성: `python src/secom_spotfire_style.py` → `results/secom_spotfire_*.png`

## ⚠️ 정직성 표기 (반드시 읽을 것)

- 이 그림들은 **Spotfire로 만든 `.dxp`가 아니라, 같은 차트 '유형'을 Python(matplotlib)으로
  재현**한 것입니다. 실제 Spotfire 대시보드 구축은 별도 트랙(WM-811K 웨이퍼맵)에서 진행합니다.
- SECOM은 **익명 센서(S0–S589) + pass/fail + 타임스탬프**뿐이라, 매뉴얼의 `Equipment_ID`,
  `Yield`, `Chamber_Pressure`, `Wafer_ID` 컬럼이 없습니다. 따라서 **가짜 설비/공정 라벨을 만들지
  않고**, 설비별 분해 대신 **pass/fail·센서 통계영향**으로 정직하게 대체했습니다.

## 차트별 설명 (매뉴얼 의도 → SECOM 적용)

| Spotfire 차트 | 매뉴얼 의도 | SECOM 적용 (정직한 대응) | 파일 |
|---|---|---|---|
| **Box Plot** | 설비 호기별 수율·두께 편차 비교 | 상위 수율영향 센서의 **pass vs fail 분포**(z-정규화) 비교 | `secom_spotfire_box.png` |
| **Scatter Plot Matrix** | 다차원 공정 파라미터 상관 스크리닝 | 상위 4개 센서 간 산점 행렬(파랑=pass/빨강=fail) | `secom_spotfire_scatter_matrix.png` |
| **Treemap** | 공정·설비 분류별 불량 분포 | 설비 분류 없음 → **센서별 통계영향(−log10 p) 크기·효과크기 음영** | `secom_spotfire_treemap.png` |
| **파라미터 매핑 산점도** | X=Process_Time, Color by Equipment_ID | X=타임스탬프, Y=상위 센서, **Color=pass/fail**(Equipment_ID 없음) | `secom_spotfire_parammap.png` |

## 읽는 법
- **Box Plot**: 두 상자(pass/fail) 위치가 어긋날수록 그 센서가 불량을 잘 가림.
- **Scatter Matrix**: 빨간 점(fail)이 특정 영역에 몰리는 축 조합 = 불량과 관련 큰 파라미터.
- **Treemap**: 큰 타일 = 통계적으로 불량과 관련 큰 센서(우선 관리 대상).
- **Parameter-map**: 시간축에서 fail(X 마커)이 특정 시기·값대에 몰리는지 = 시계열 이상.

## 왜 이렇게 했나 (직무 연계)
양산기술의 데이터 분석 우대역량(SPC·상관분석·시각화 기반 스크리닝)을, **현업 도구(Spotfire)가
제공하는 차트 유형의 의도를 이해하고 동일 분석을 코드로 재현**할 수 있음을 보이기 위함. 실제
Spotfire 조작 숙련은 WM-811K 트랙에서 `.dxp`로 별도 증명 예정.
