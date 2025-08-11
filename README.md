
README — 工业数据异常检测项目

⸻

1. 项目简介 (Project Overview / 프로젝트 소개)

中文：
本项目是针对工业数据的异常检测任务所设计的 Python 实现。前一版本可能因为数据量不足（约100条样本）导致模型在检测“异常”类别时准确率为0。为解决此问题，本版本在数据规模、特征处理及模型结构等方面进行了优化，提升了模型的泛化能力与异常检测性能。

English:
This project is a Python-based industrial data anomaly detection system. The previous version may have had insufficient data (around 100 samples), resulting in 0 accuracy for detecting the “Anomaly” class. To address this, this upgraded version improves dataset size, feature handling, and model architecture to enhance generalization and anomaly detection performance.

한국어:
이 프로젝트는 산업 데이터 이상 탐지를 위한 Python 구현입니다. 이전 버전은 데이터가 약 100개 정도로 부족하여 ‘이상(Anomaly)’ 클래스 탐지 정확도가 0%였습니다. 이를 해결하기 위해 이번 버전에서는 데이터 규모, 특성 처리, 모델 구조 등을 개선하여 일반화 성능과 이상 탐지 성능을 향상시켰습니다.

⸻

2. 数据说明 (Dataset Description / 데이터 설명)

中文：
    •    数据规模：1000条样本
    •    数据特征：包含多种数值型传感器数据字段
    •    标签类别：正常（Normal）与异常（Anomaly）二分类
    •    数据来源：工业生产环境模拟生成（无真实敏感数据）

English:
    •    Size: 1,000 samples
    •    Features: Multiple numerical sensor data fields
    •    Target classes: Binary classification — Normal / Anomaly
    •    Source: Simulated industrial production environment (no sensitive real data)

한국어:
    •    데이터 규모: 1,000개 샘플
    •    데이터 특성: 다양한 수치형 센서 데이터 필드 포함
    •    라벨 클래스: 정상(Normal) / 이상(Anomaly) 이진 분류
    •    데이터 출처: 산업 생산 환경 시뮬레이션 생성 (민감한 실제 데이터 아님)

⸻

3. 改进的地方 (Improvements / 개선 사항)

中文：
    1.    数据规模扩充：将样本量从100条提升至1000条，提高模型训练的稳定性与鲁棒性。
    2.    模型多样化：在随机森林（Random Forest）基础上新增支持向量机（SVM），提升不同算法下的检测效果。
    3.    数据预处理优化：引入箱线图与IQR方法检测并处理异常值，用中位数替换异常值，减少噪声影响。

English:
    1.    Increased dataset size: Expanded from 100 to 1,000 samples for more stable and robust training.
    2.    Model diversification: Added Support Vector Machine (SVM) alongside Random Forest to enhance detection across different algorithms.
    3.    Optimized preprocessing: Used boxplot and IQR method for outlier detection and replaced anomalies with median values to reduce noise impact.

한국어:
    1.    데이터 규모 확장: 샘플 수를 100개에서 1,000개로 늘려 모델 학습의 안정성과 강건성을 향상.
    2.    모델 다양화: 랜덤 포레스트(Random Forest) 외에 SVM(서포트 벡터 머신)을 추가하여 다양한 알고리즘 기반의 탐지 성능 향상.
    3.    데이터 전처리 최적화: 박스플롯(Boxplot)과 IQR 방법을 사용해 이상치를 탐지하고, 중앙값으로 대체하여 노이즈 영향을 줄임.

⸻

4. 模型流程 (Model Workflow / 모델 흐름)

中文：
    1.    数据读取与特征选择
    2.    异常值检测与处理（IQR + 中位数替换）
    3.    数据集划分（训练集 / 测试集）
    4.    模型训练（随机森林 + SVM）
    5.    模型评估（准确率、分类报告、混淆矩阵）

English:
    1.    Data loading and feature selection
    2.    Outlier detection and handling (IQR + median replacement)
    3.    Train-test split
    4.    Model training (Random Forest + SVM)
    5.    Model evaluation (Accuracy, classification report, confusion matrix)

한국어:
    1.    데이터 로딩 및 특성 선택
    2.    이상치 탐지 및 처리 (IQR + 중앙값 대체)
    3.    학습용 / 테스트용 데이터 분할
    4.    모델 학습 (랜덤 포레스트 + SVM)
    5.    모델 평가 (정확도, 분류 보고서, 혼동 행렬)

⸻

5. 项目结果 (Results / 프로젝트 결과)

中文：
    •    数据集规模扩大后，模型在“异常”类别的检测效果明显提升。
    •    随机森林模型准确率达0.96，支持向量机模型准确率为0.82。
    •    随机森林在异常类别上表现更优（精准率0.98，召回率1.00，F1分数0.90），支持向量机表现较弱（精准率0.35，召回率0.44，F1分数0.44）。
    •    异常值处理显著减少了误判与漏判的概率。

English:
    •    With a larger dataset, detection performance for the “Anomaly” class improved significantly.
    •    Random Forest model accuracy reached 0.96, while SVM model accuracy was 0.82.
    •    Random Forest outperformed SVM on the anomaly class (precision 0.98, recall 1.00, F1-score 0.90), while SVM showed weaker performance (precision 0.35, recall 0.44, F1-score 0.44).
    •    Outlier handling greatly reduced false positives and false negatives.

한국어:
    •    데이터셋 확장 후 ‘이상’ 클래스의 탐지 성능이 크게 향상됨.
    •    랜덤 포레스트 모델 정확도는 0.96, SVM 모델 정확도는 0.82임.
    •    랜덤 포레스트가 이상 클래스에서 더 우수한 성능(정밀도 0.98, 재현율 1.00, F1 점수 0.90)을 보였고, SVM은 상대적으로 낮은 성능(정밀도 0.35, 재현율 0.44, F1 점수 0.44)을 나타냄.
    •    이상치 처리를 통해 오탐 및 미탐 가능성을 크게 줄임.

⸻

6. 后续改进方向 (Future Improvements / 향후 개선 방향)

中文：
    •    引入更多样化的异常检测算法，如神经网络、自动编码器（Autoencoder）等，提升模型的检测能力。
    •    结合时间序列分析，利用工业设备的时间动态特征进行更精准的异常识别。
    •    增加特征工程步骤，尝试提取更丰富的特征以提升模型性能。
    •    开发模型自动调参与集成方法，提高模型的泛化和稳定性。
    •    完善数据采集与预处理流程，实现实时数据监控和在线异常检测。

English:
    •    Introduce more diverse anomaly detection algorithms such as neural networks and autoencoders to improve detection capability.
    •    Incorporate time series analysis to leverage temporal dynamics of industrial equipment for more accurate anomaly detection.
    •    Enhance feature engineering by extracting richer features to boost model performance.
    •    Develop automated hyperparameter tuning and ensemble methods to improve generalization and stability.
    •    Improve data acquisition and preprocessing pipeline for real-time monitoring and online anomaly detection.

한국어:
    •    신경망 및 오토인코더(Autoencoder)와 같은 다양한 이상 탐지 알고리즘 도입으로 탐지 성능 향상.
    •    산업 장비의 시계열 특성을 활용한 시간 시계열 분석 적용으로 더 정밀한 이상 탐지 구현.
    •    보다 풍부한 특성 추출을 통한 특성 엔지니어링 강화로 모델 성능 향상.
    •    자동 하이퍼파라미터 튜닝 및 앙상블 기법 개발로 일반화 및 안정성 증대.
    •    실시간 데이터 모니터링과 온라인 이상 탐지를 위한 데이터 수집 및 전처리 프로세스 개선.
