Spatiotemporal Prediction of Monthly Precipitation: A systematic Review of  Hybrid Models

Manuel, R, Pérez ![](Review_20250912_PerezManuel_assets/image_16.png)<sup>a</sup>, Marco, J, Suárez ![](Review_20250912_PerezManuel_assets/image_16.png)<sup>b</sup>, Oscar, J, García ![](Review_20250912_PerezManuel_assets/image_16.png)<sup>c</sup>

<sup>a</sup>Doctoral Program in Engineering, Pedagogical and Technological University of Colombia, Sogamoso 152210, Colombia,                                 <sup>b</sup>School of Systems and Computing Engineering, Pedagogical and Technological University of Colombia, Sogamoso 152210, Colombia	         <sup>c</sup>School of Geological Engineering, Pedagogical and Technological University of Colombia, Sogamoso 152210, Colombia,                       *Corresponding author. E-mail:

![](Review_20250912_PerezManuel_assets/image_16.png)MP, ; MS, ; OG,

## **______****_**

## **ABSTRACT**

Hybrid and ensemble machine learning models are increasingly used to predict monthly precipitation; however, their true value depends on their integration with hydrologic data realities and physics-based modeling. Eighty-five studies (2020–2025) were systematically reviewed, and a quantitative synthesis was conducted comparing hybrid classes—decomposition-based, optimization-enhanced, component combination/ensemble, and post-processing hybrids—against non-hybrid baselines. Effect sizes defined as log ratios of RMSE/MAE indicate that decomposition-based hybrids and stacking/averaging schemes deliver the most consistent improvements (typical RMSE reductions ≈15–35% across studies), while probabilistic post-processing remains underused at monthly horizons. The review clarifies how ML complements, rather than replaces, numerical weather prediction—chiefly in downscaling and bias correction/post processing, and as fast surrogates. Recurrent pitfalls are highlighted (data leakage, station non-representativeness, and non-stationarity). The study provides a unified taxonomy, reproducible synthesis templates, and practical recommendations for robust monthly precipitation prediction.

**Key Words:** Hybrid models, machine learning, monthly precipitation prediction, spatiotemporal prediction.

## **_****_****______**

## **HIGHLIGHTS**

1. A systematic review of hybrid models for monthly precipitation prediction with a spatiotemporal focus is presented.

1. Classification of hybridization approaches based on preprocessing, parameter optimization, component combination, and postprocessing stages.

1. The tables included in the study offer a detailed synthesis of the reviewed hybrid models, their configurations, evaluation metrics, and geographic areas of application.

1. A comprehensive performance improvement analysis was conducted for the top two models from each study, with a percentage comparison of their respective accuracy. A log-transformed RMSE and MAE graph is presented, offering a clear visualization of the models' performance in predicting monthly precipitation.

## **_________**

## GRAPHICAL ABSTRACT

![](Review_20250912_PerezManuel_assets/image_05.jpg)![](Review_20250912_PerezManuel_assets/image_01.jpg)![](Review_20250912_PerezManuel_assets/image_14.png)![](Review_20250912_PerezManuel_assets/image_08.png)![](Review_20250912_PerezManuel_assets/image_03.png)![](Review_20250912_PerezManuel_assets/image_17.png)

## **______________****_**

1. Introduction

Precipitation, derived from the condensation of atmospheric water vapor, is the primary source of freshwater on Earth. It includes forms like drizzle, rain, sleet, snow, and hail. With less than 1% of Earth’s water being fresh and accessible, mainly replenished through precipitation, accurate prediction is essential for effective water resource management . In recent years, climate change has intensified the challenges associated with precipitation prediction by altering global patterns, making dry regions drier in subtropical zones and wet regions wetter in mid to high latitude areas . These shifts significantly impact agriculture, the economy, water supply, and energy production, highlighting the need for effective predictive modeling to mitigate potential impacts .

In response to these challenges, machine learning (ML) has emerged as a powerful tool, enabling computers to detect patterns and excel in prediction and classification tasks. Among its subfields, Deep Learning (DL) an advanced form of Artificial Neural Networks has demonstrated particular effectiveness in modeling complex, nonlinear systems. These models simulate brain-like processes to handle data across various levels of abstraction (Rahman *et al.*, 2022; Anwar *et al.*, 2018;. DL techniques have been successfully applied to precipitation forecasting across daily, weekly, monthly, and annual timescales, supporting applications in flood and landslide prediction, as well as in water supply and quality management .

Traditional statistical models, which explore the relationship between precipitation and variables such as humidity, temperature, wind speed, and atmospheric pressure, often fall short due to the inherent complexity of hydrological systems and the influence of geomorphological and climatic variability (Barrera-Animas et al., 2022; Wu and Chau, 2013; Li et al., 2023). This has led to the increased application of ML models, such as backpropagation neural networks , Long short-term memory (LSTM) networks , random forests , and support vector machines (SVM) some enhanced with generic algorithms for parameter optimization ; . These models are commonly evaluated using metrics like RMSE, MAE and NSE to ensure prediction reliability .

Among the most widely applied techniques, Artificial Neural Networks (ANNs) have become essential tools for precipitation forecasting across timescales. For instance,  used deep neural networks to estimate precipitation from remote sensing data, demonstrating their effectiveness in handling complex datasets. Likewise,  integrated convolutional neural networks, LSTM, and attention mechanisms to predict monthly precipitation using CHIRPS data at high spatial (0.05° * 0.05°) and temporal resolution, achieving superior RMSE and MAE results compared to Gated Recurrent Unit (GRU) and LSTM-AT models. Further research on ANN applications includes monsoon rainfall prediction in India using backpropagation, which significantly reduced prediction errors compared to traditional methods . In Kerala, an Adaptive Basis Function Neural Network (AB-FNN) a backpropagation algorithm variation outperformed fourier analysis, achieving a stabilized mean squared error of 0.085 during training .

In addition, soft computing techniques applied during the Indian monsoon, such as feedforward neural networks (FFNNs) and multilayer perceptrons (MLPs), reduced prediction errors from 18.3% to 10.2% relative to persistence models, indicating substantial gains in accuracy .

The STConvS2S model proposed by , which uses 3D convolutional neural networks within a sequence-to-sequence framework, has shown a 23% improvement in predicting future sequences and operates five times faster than RNN- based models. By leveraging a sequence-to-sequence network based on 3D convolutional neural networks, excels in climate prediction, including daily precipitation. It demonstrates a 23% improvement in predicting future sequences, operating five times faster than RNN-based models. Similarly,  demonstrated that a hybrid model combining ANN with backpropagation and generic algorithms significantly reduced mean squared error and improved the coefficient of determination compared to standard ANN models.

Other innovative approaches include focused time-delay neural network models , which dynamically adapt to temporal data by optimizing delay times through trial and error. These models demonstrate high accuracy for annual precipitation, although performance declines at shorter temporal resolutions. Modular ANNs incorporating preprocessing techniques such as Moving Averages (MA), Principal Component Analysis (PCA), and Singular Spectrum Analysis (SSA) have outperformed conventional models, yielding superior performance in terms of Coefficient of Efficiency (CE), RMSE, and Prediction Intervals (PI)  .

Modular systems that combine support vector regression (SVR) with ANN have also improved accuracy for stations like Zhenwan and Wuxi . Meanwhile  applied ANN for downscaling and evaluating satellite-based precipitation estimates, leveraging Particle Swarm Optimization (PSO), imperialist competitive algorithm (ICA), and Genetic Algorithm (GA) to enhance performance, demonstrating improved prediction quality.

Recent advancements have focused on hybrid models that integrate time series decomposition techniques with machine learning algorithms. For instance,  showed that combining Complementary Ensemble Empirical Mode Decomposition (CEEMD) with wavelets significantly improves forecasting at annual, monthly, and daily levels. Ensemble methods such as bagging, boosting, and stacking have further enhanced prediction accuracy by aggregating model outputs through a meta-learner . The CEEMD-FCMSE (Fuzzy Comprehensive Multi-Criteria Evaluation)-Stacking model, for instance, significantly reduces error metrics, demonstrating the superior capabilities of such advanced models .

A detailed assessment of hybrid and ensemble models is therefore essential to better understand their advantages and limitations. This systematic review consolidates recent developments in hybrid precipitation forecasting by analyzing the integration of ANNs, random forests, and SVMs with statistical methods. It highlights the substantial error reductions achieved by incorporating CEEMD, SSA, wavelet transforms, and various optimization algorithms compared to standalone models . Moreover, ensemble strategies such as model averaging, stacking, bagging, boosting, and dagging have further improved model robustness and reliability . Unlike prior reviews, this study offers a more comprehensive systematic analysis by explicitly comparing specific hybrid approaches based on key performance metrics and identifying critical knowledge gaps. These contributions lay the groundwork for future research aimed at addressing the growing challenges posed by climate change in hydrological prediction.

## 1.2 Hydrologic data & modeling context

Monthly precipitation prediction operates at the interface of hydrologic data limitations and climate dynamics. Station networks are sparse and biased toward populated valleys; gauges in complex terrain undersample windward–leeward contrasts. Satellite/gridded products (e.g., CHIRPS, ERA5) alleviate gaps but introduce retrieval and orographic biases, scale mismatches, and temporal discontinuities . These factors yield heteroskedastic, spatially autocorrelated targets. As a result, hybrid designs that explicitly address spatial structure (e.g., elevation aware clustering, spatial cross validation; , and temporal multi-scale variability (decomposition + learning) are especially relevant at monthly horizons. Moreover, the role of physics is central: large-scale circulation constrains anomalies, while land–atmosphere feedbacks modulate local totals—hybrids should respect these constraints via cautious feature design, leakage-safe validation, and post-processing that calibrates distributions .

## 1.3 Objectives and contributions

1. Taxonomy: Provide a clear functional taxonomy of hybrid monthly precipitation predictors (preprocessing, optimization, component combination/ensemble, post-processing).

1. Quantitative synthesis: Derive cross-study effect sizes (log RMSE/MAE ratios vs best non-hybrid baseline) and compare hybrid classes.

1. Practice-oriented guidance: Identify validation pitfalls and provide leakage-safe, spatiotemporal CV recommendations for monthly horizons.

1. Positioning with physics: Clarify how ML hybrids complement physical NWP (downscaling, bias correction/post processing, surrogates) and highlight underexplored probabilistic hybrids.

## ---

## MATERIALS AND METHODS

This section describes the stages of the systematic review using the PRISMA methodology . A literature search on hybrid and ensemble ML (machine learning) for precipitation prediction was conducted in major databases, including Scopus and Science Direct. The systematic search strategy was developed using the ScienceDirect, Scopus, and IEEE Xplore databases. The search was conducted between January 1, 2020, and March 31, 2025. Only articles published in English or Spanish in peer-reviewed journals were included. Both original research articles and review papers were considered. The Boolean search combinations used were:

1. (“machine learning” AND precipitation AND monthly AND hybrid)

1. (“machine learning” AND precipitation AND monthly AND hybrid AND spatiotemporal)

1. (“hybrid model” AND prediction AND precipitation AND wavelet)

1. (“hybrid model” AND prediction AND precipitation AND LSTM)

The results were filtered to include only articles with full-text availability, and open access articles were prioritized whenever possible. Technical reports, book chapters, preprints, and grey literature were excluded. The selection process and eligibility criteria are presented in Table 1 and Figure 1. At the outset, 452 articles were identified, of which 130 were duplicates and subsequently. After reviewing titles and abstracts, 160 articles were excluded for not meeting the criteria. Finally, 162 articles were assessed in full text and 85 were included in the final analysis, as detailed in Figure 1. ScienceDirect offers critical environmental and hydrological journals such as Journal of Hydrology, Environmental Modelling & Software, and Expert Systems with Applications, which publish foundational studies on hybrid modeling approaches (Ardabili et al., 2020; Wang et al., 2024). Scopus was chosen for its comprehensive interdisciplinary indexing, including prominent journals like Water Resources Management, Hydrological Sciences Journal, and Remote Sensing, renowned for publishing innovative precipitation modeling research (Ali et al., 2020; Kumar et al., 2021). IEEE Xplore provides specialized coverage in computational intelligence and machine learning, indexing influential IEEE journals such as IEEE Transactions on Neural Networks and Learning Systems and IEEE Transactions on Geoscience and Remote Sensing, essential for advanced studies on hybrid and ensemble precipitation prediction models (Castro et al., 2021; Tang et al., 2022). Together, these databases ensured a rigorous and comprehensive capture of relevant, high-impact literature, significantly enhancing the depth and credibility of this systematic review. Figure 1 shows the identification stage. In this stage, a preliminary analysis was conducted to establish the information sources and search strategies, and the eligibility and inclusion criteria were defined. The first stage included searching for information sources. Subsequently, duplicate articles were removed, the snowball technique is applied to identify additional studies, and the duplicate articles were filtered out completely. The screening stage was divided into two reviews. The first review examined the title, abstract, and keywords to determine whether the article met the inclusion and exclusion criteria. The selected articles were then reviewed again, considering the introduction and conclusions. Subsequently, in the eligibility stage, (Figure 1), a full review of the articles selected during the screening stage was conducted. In the eligibility stage, trends and opportunities were carefully observed. Finally, the set of articles and research studies to be included in the review was obtained.

![](Review_20250912_PerezManuel_assets/image_05.jpg)

Figure 1 | PRISMA flow diagram for study selection. The diagram shows the number of records identified, excluded, and ultimately included in the review.

## 2.1 Literature Analysis before Evaluation

This section defines the scope of the systematic review and establishes the inclusion and exclusion criteria. It also develops a detailed search strategy, including keywords and time ranges. This preliminary analysis is essential to ensure a solid, well-defined foundation that guarantees the process is systematic, comprehensive, and replicable. A series of central themes are proposed among the research limitations, which will specify the searches for the closest research studies related to this doctoral proposal.

The study identifies five key keywords that represent central themes of the research: Machine Learning, Precipitation, Monthly, Hybrid, and Spatiotemporal. Each of these terms encapsulates essential aspects of the study's focus. Machine Learning is highlighted due to its foundational role in the analytical approach, while Precipitation represents a core meteorological variable under investigation. The keyword Monthly reflects the temporal resolution used in the data analysis. The term Hybrid denotes the integration of multiple methods or models, and Spatiotemporal emphasizes the study’s focus on variations across both space and time. All of these keywords are not only frequently used but also serve as the conceptual pillars of the research.

The limitations of the study were organized into three exclusion criteria that help define the scope and focus of the analysis. The first criterion (CE1) excludes extreme events, as these are beyond the intended scope of the research. The second criterion (CE2) refers to the omission of other geographical regions and phenomena that are not considered within the proposal. Finally, the third criterion (CE3) excludes topics that are explicitly outside the objectives of the study. These classification criteria facilitate the filtering of information and the delineation of relevant content.

A wide range of topics were excluded from the scope of the study as part of the established exclusion criteria. Under CE2, excluded themes include typhoons, cyclones, monsoons, drought, heavy rainfall, extreme rainfall, slope failures, standardized precipitation indices, data intelligence models, and other geographical phenomena such as debris, air pollution, and aquifer levels. CE3 groups exclusions that are beyond the study’s objective, such as landscape, runoff, streamflow, groundwater, flood, wildfire, water footprint, and rainfall. This structured classification ensured a focused and coherent research framework by eliminating topics not directly aligned with the study’s purpose.

## 2.2 Literature Analysis after Evaluation

The systematic review of bibliographic sources followed the guidelines proposed by  , selecting two bibliographic databases and a subsequent document filtering process through identification, screening, eligibility, and inclusion. The snowball technique was applied using the software VOS Viewer, which adds search results from databases to obtain data views that improved article searches with terms not initially considered but highly valuable for the ongoing research. Then, duplicates were excluded in both databases for the original searches and the snowball method. The next step involved screening, where articles were filtered by reviewing titles, abstracts, introductions, and conclusions and where the inclusion criteria.

## (1)

## (2)

The VOSviewer analysis included a variety of terms categorized according to their conceptual and computational approaches, along with the time of their appearance. Terms such as data, model, and rainfall were associated with general, machine learning (ML), and climatic approaches, all appearing early in the timeline (mostly in 2022). Metric-based terms like MAE and RMSE were classified under the metric approach and emerged in mid to late 2022 and early 2023, respectively. More advanced machine learning terms, such as LSTM, prediction, and wavelet, corresponded to higher computational levels, particularly levels 2 and 3, reflecting increasing complexity in the methodology. The temporal distribution of these terms, from early 2022 to early 2023, demonstrates a progressive refinement in the use of analytical tools within the study.

    By filtering only by the study focus on ML techniques and precipitation, a subset of key terms was identified. Terms such as hybrid model and model were classified under ML approaches and grouped into category 1, both appearing in early 2022. More advanced ML techniques including LSTM, prediction, and *wavelet* were placed in groups 2 and 3, also within the same year, indicating a progression in methodological complexity. The term precipitation was categorized under the climatic approach and assigned to group 2, reinforcing its relevance as a climatic variable within the scope of this filtered analysis. This focused selection highlights the thematic convergence of machine learning methodologies with precipitation-centered studies in the given timeframe. New search conditions could be added for the databases based on the terms found at the computational level and machine learning algorithms and this study's central climatic variable, precipitation. To improve searches, conditions focused specifically on the words extracted in Table 1 focusing the searches on hybrid model by omitting the word model as it is compound and performing separate searches for the words Wavelet and LSTM as they are not complementary, with data window of 2020 to 2025 and research and review type articles that are open access.

    Methodological quality criteria were applied to assess the risk of bias in the included studies, based on the robustness of the reported validations and the transparency in model descriptions. Studies with insufficient information were marked for qualitative rather than quantitative analysis.

Table 1 | Search Query Results for Hybrid Precipitation Prediction Models Using Wavelet and LSTM, Input for VOSviewer Analysis.

| Title, abstract, keywords | Source | Result |
| --- | --- | --- |
| hybrid model AND prediction AND precipitation AND wavelet | Science Direct | 6 |
| hybrid model AND prediction AND precipitation AND lstm | Science Direct | 35 |
| hybrid model AND prediction AND precipitation AND wavelet | Scopus | 18 |
| hybrid model AND prediction AND precipitation AND lstm | Scopus | 17 |
| ("All Metadata":hybrid mode) AND ("All Metadata":prediction) AND ("All Metadata":precipitation) AND ("All Metadata":wavelet) | IEEE | 0 |
| ("All Metadata":hybrid mode) AND ("All Metadata":prediction) AND ("All Metadata":precipitation) AND ("All Metadata":lstm) | IEEE | 1 |
|  |  |  |
|  |  |  |

## 2.3 Quantitative synthesis protocol

For each study and metric (RMSE, MAE), the hybrid error , and the best non-hybrid baseline error  at monthly resolution was extracted. The effect size is computed as the log-ratio, as shown in Equation (3), where negative values indicate improvement for “lower-is-better” metrics. Values of Delta are summarized by hybrid class using medians and IQRs, and differences between classes are evaluated via a Kruskal–Wallis test with post-hoc Dunn tests (alpha = 0.05). Where studies report R²/NSE only, metrics are converted to error proxies when possible or analyzed separately. All extraction sheets and code templates are provided for reproducibility. Across-study pooling that would require sample-size weighting is avoided, given heterogeneous designs; robust nonparametric summaries are emphasized; benchmarking/verification guidance follows , and .

## (3)

## ---

1. Hybrid Predictive Models

Hybrid predictive models combine two or more machine learning or soft computing methods to achieve better performance by leveraging each method's advantages and purposes . Often, one method is used for prediction and another to optimize that prediction, functioning like a company with employees from diverse specialties working towards a common goal. This approach produces more robust models, improving accuracy and surpassing the performance of individual models by capitalizing on the strengths of each component. It encompasses all computational phases, from data normalization to decision-making . A categorization proposed by  for hybrid models applied to time series, potentially generalizable to other contexts, identifies four main components:

![](Review_20250912_PerezManuel_assets/image_17.png)

Figure 2 | Classification of hybrid modeling approaches based on functional integration.

Figure 2 presents a classification of hybrid modeling approaches into four main categories, each corresponding to a different stage of the modeling process. First, data preprocessing-based hybrid models aim to improve input quality through techniques such as normalization, decomposition, or noise reduction. Second, parameter optimization-based hybrid models employ metaheuristic algorithms (such as PSO, GA, or HBA Honey Badger Algorithm) to fine-tune the hyperparameters of base models, enhancing their performance. The third category, component combination-based hybrid models, focuses on integrating different parts of multiple models—for example, combining the predictive structure of a statistical model with the learning capabilities of a neural network. Finally, postprocessing-based hybrid models operate on the model outputs to correct errors or refine predictions, using techniques like calibration or ensemble post-adjustments.

## 3.1 Data Preprocessing-based Hybrid Models

Data preprocessing refers to a set of techniques applied to raw data prior to modeling, with the goal of improving data quality, consistency, and analytical value. These techniques may involve normalization, outlier removal, missing value imputation, feature scaling, or more complex transformations such as Principal Component Analysis (PCA) or clustering. By aligning data distributions, reducing dimensionality, and eliminating noise, preprocessing lays the foundation for more robust and accurate predictive modeling. In contrast, data preprocessing-based hybrid models represent a specific modeling architecture in which the preprocessing stage is explicitly integrated as a core component of a hybrid framework. Rather than treating preprocessing as an independent or auxiliary step, these hybrid models are designed to combine preprocessing techniques with one or more learning algorithms in a structured and interdependent manner. The goal is to not only clean or transform data, but to reshape the input space in ways that enhance the learning capacity of the predictive engine.

For example, clustering meteorological stations using K-means into climatically homogeneous regions does not modify raw precipitation values but restructures the input feature space. When such preprocessing is tightly coupled with machine learning models e.g., Support Vector Regression (SVR) or Group Method of Data Handling (GMDH) the resulting hybrid model performs better in handling spatial heterogeneity and temporal variability. Empirical evidence indicates that hybrid model configurations incorporating preprocessing, such as residual decomposition coupled with machine learning algorithms like Support Vector Regression (SVR), Gene Expression Programming (GEP), and the Group Method of Data Handling (GMDH), can substantially enhance forecasting accuracy. Specifically, studies report up to a 67% reduction in Mean Square Error (MSE) and a 5% improvement in Nash-Sutcliffe Efficiency when such hybrid models are further optimized using genetic algorithms. Moreover, the application of Savitzky-Golay filtering as a noise-reduction preprocessing step has been shown to increase prediction accuracy in daily rainfall estimation via Adaptative Neuro Fuzzy Interference System – Artificial Bee Colony (ANFIS-ABC) models, with the correlation coefficient (R) rising from 0.78 to 0.85 .

![](Review_20250912_PerezManuel_assets/image_09.png)

Figure 3 | Classification of hybrid models based on data preprocessing. The main methods used in the preprocessing stage are illustrated, along with their impact on the predictive performance of hybrid models.

The following figure is a classification of hybrid models based on data preprocessing techniques, emphasizing methods designed to enhance data quality before predictive modeling. This classification includes Principal Component (PCA), which effectively reduces data dimensionality; Canonical Factor Analysis (CFA) and the Autocorrelation Function (ACF), both of which help identify statistical relationships between temporal variables; and Singular Spectrum Analysis (SSA), a powerful tool for decomposing complex time series. The K-means algorithm is also widely used as a method for unsupervised data clustering. Moreover, Figure 3 highlights the use of sequential preprocessing strategies, where multiple techniques are applied in series to extract different latent structures or patterns from the data.

Furthermore, the approach of sequential preprocessing combinations is highlighted, allowing multiple techniques to be applied in series to extract various hidden patterns or structures within the data. For instance,  employed the K-means algorithm to enhance data structure through clustering, while  utilized the ACF to analyze temporal dependencies within the rainfall time series.

## **3.2 ****Parameter ****O****ptimization-based ****H****ybrid ****M****odels****.**

![](Review_20250912_PerezManuel_assets/image_19.png)

Figure 4  | Optimization algorithms commonly used in parameter optimization-based hybrid models.

These hybrid models incorporate a systematic optimization layer that autonomously tunes hyperparameters and internal configurations of forecasting algorithms. Unlike traditional grid or manual search, modern hybrid systems utilize metaheuristic algorithms, such as (GA), (PSO), and Ant Colony Optimization (ACO), to effectively explore high-dimensional parameter spaces .

In recent applications, GA has been employed to optimize the weight combination of (Gene Expression Programming) GEP, SVR, and GMDH outputs, leading to drastic improvements in model accuracy. At the Rasht station, this strategy reduced MSE by 67% and increased forecast skill significantly compared to SVR-based hybrids . Furthermore, novel techniques such as the Brownian Motion-based Pelican Optimization Algorithm (BMPOA) have demonstrated superior performance in fine-tuning hybrid Deep Belief Networks (DBNs) and (LSTM) models .

The diagram (Figure 4 ) illustrates the optimization algorithms employed to enhance the performance of predictive models. These algorithms, predominantly metaheuristic and bio-inspired in nature, are designed to fine-tune the hyperparameters of base models, enabling the identification of optimal configurations that maximize system accuracy. Bio-inspired algorithms draw inspiration from natural processes and behaviors, allowing them to efficiently explore complex solution spaces. Among the widely used methods are Particle Swarm Optimization (PSO), which simulates the social behavior of bird flocking to guide search agents; Genetic Algorithm (GA), based on the principles of natural selection and genetics; Bat Algorithm (BAT), inspired by the echolocation behavior of bats; and Honey Badger Algorithm (HBA), which mimics the foraging and digging behavior of honey badgers. Additionally, the diagram includes less conventional techniques such as Biogeography-Based Optimization (BBO), which models the migration patterns of species across habitats; Fruit Fly Optimization Algorithm (FFOA), based on the food-searching behavior of fruit flies; Firefly Algorithm (FFA), inspired by the flashing patterns and attraction mechanisms of fireflies; Chimp Optimization Algorithm (CPO), which emulates the intelligent hunting strategies of chimpanzees; and Shuffled Complex Evolution (SCE), a population-based algorithm that combines complex shuffling and evolutionary principles to enhance convergence.

These algorithms have demonstrated their effectiveness in recent studies. For instance, in the article *”*, techniques such as PSO and GA to optimize artificial intelligence models for rainfall prediction, resulting in notable improvements in accuracy. Similarly, the study *“**”* implemented several metaheuristic algorithms, including HBA, to optimize both deep neural networks and Extreme Learning Machines (ELM), achieving superior performance in long-term rainfall forecasting.

## **3.3 ****Component ****C****ombination-based ****H****ybrid ****M****odels**

Component combination-based hybrid models integrate multiple forecasting algorithms into a unified framework to exploit their complementary strengths. Each component is designed to capture distinct statistical or dynamic patterns within the data, thereby improving the model’s overall robustness and accuracy (Figure 5).

These models frequently employ ensemble learning techniques such as Bagging, Boosting, and Stacking—to combine the outputs of diverse base learners. In Bagging (e.g., Random Forests), models are trained in parallel on bootstrapped subsets, and predictions are aggregated (usually via voting or averaging), which helps reduce variance. Boosting, by contrast, trains models sequentially, where each new learner corrects the residuals of its predecessors, thereby reducing bias . Stacking goes a step further: it combines the predictions of multiple base models using a meta-learner, which is trained to optimize final predictions based on the base models’ outputs, often improving performance on complex nonlinear tasks .

This hybrid architecture has shown significant improvements in precipitation forecasting, especially under noisy, multiscale, or nonstationary conditions. For instance, integrating decision trees, SVR, and ANN into a single ensemble has yielded lower RMSE values and better bias calibration compared to individual models . Furthermore, hybrid frameworks integrating Ant Colony Optimization (ACO) with Gradient Boosted Decision Trees (GBDT) have achieved prediction accuracies exceeding 93% for monthly precipitation forecasts, surpassing the performance of either component alone .

Further expanding on ensemble predictive models, these combine multiple training algorithms to improve predictive accuracy, robustness, and overall performance. By aggregating the strengths of diverse models, these methods reduce errors and enhance generalization. Ensemble techniques are widely regarded as a key approach within supervised learning frameworks, offering significant advantages in handling complex datasets and achieving superior results . There are five main techniques for developing ensemble models: model averaging, stacking, bagging, boosting, and dagging .

![](Review_20250912_PerezManuel_assets/image_15.png)

Figure 5 | Taxonomy of component combination-based hybrid modeling structure.

## **3.4 ****Model ****A****veraging**

This method involves training multiple models independently and averaging their predictions to reduce variance and improve generalization .

For models , the averaged prediction  is:

## (3)

The Model Averaging (MA) approach has become increasingly popular as a straightforward and effective Ensemble Machine Learning (EML) method for hydrological process modeling. This approach encompasses a variety of techniques, including Simple Arithmetic Mean (SAM), Bayesian Model Averaging (BMA), Equal Weights Averaging (EWA), Bates-Granger Model Averaging (BGA), Averaging based on Akaike’s Information Criterion (AICA), Bayes’ Information Criterion (BICA), Mallows Model Averaging (MMA), and Granger-Ramanathan Averaging (GRA) . BMA is an exceptionally efficient and widely recognized algorithm because it addresses model uncertainty. In BMA, predictions are generated through a weighted averaging process, where the weights are derived from the posterior probabilities of the competing models. Below is a concise explanation of the BMA algorithm ,  which is a set k of models; the posterior probability (ρ) for the model  is defined as the dataset D:

## (4)

Where:

## (5)

Where D denotes the observed dataset  likelihood of model prediction, ,  the prior density of the vector parameter, and the likelihood.

## **3.5 ****Bagging (Bootstrap Aggregating)**

Bagging involves creating multiple subsets of the training data through bootstrapping (random sampling with replacement), training a model on each subgroup, and aggregating their predictions. This technique reduces variance and helps prevent overfitting ; similar to averaging, the aggregated prediction is:

## (6)

## **3.6 ****Stacking**

Stacking combines multiple models by training a meta-model (or second-level model) to make final predictions based on the outputs of base models. This approach leverages the strengths of diverse models to improve overall performance [15]. If   our base model predictions, the meta-model g predicts is:

## (7)

## **3.7 ****Boosting**

Boosting trains models sequentially, with each new model focusing on correcting the errors of its predecessors. By adjusting the weights of misclassified instances, boosting aims to reduce bias and build a strong predictive model from weak learners . The combined model is:

## (8)

Where represents the weight assigned to the model  based on its accuracy.

## **3.8 ****Dagging (Disjoint Aggregating)**

Dagging involves partitioning the training data into disjoint subsets, training a model on each subgroup, and aggregating their predictions. This method aims to reduce variance and improve model robustness :

## (9)

The applications of ensemble methods are diverse; for example,  presents a probabilistic precipitation forecasting scheme using an ensemble model (EM) based on CEEMDAN and AM-MCMC. First, precipitation series are decomposed using signal decomposition techniques (CEEMDAN). Then, empirical models (TSAM, GSM, and LSTM) produce quantitative forecasts. Finally, an ensemble model combines these forecasts with weights determined by AM-MCMC. The CEEMDAN-EM-AM scheme provides more accurate predictions and reduces uncertainty through 90% confidence intervals. Compared to individual models, the ensemble model shows greater accuracy.

In their study,  evaluated six individual and ensemble models using the stacking technique. The ensemble model outperformed the individual models, achieving an RMSE of 17.5 mm. The model integrated a decision tree, KNN, and LSTM as the base learners, with XGBoost as the second-level learner.

Detailed taxonomy of component combination-based hybrid models, which integrate multiple techniques or sub-models to enhance predictive performance. This category is subdivided according to the nature of the components being combined. For instance, decomposition-based hybrids apply transformations such as Empirical Mode Decomposition (EMD) or Discrete Wavelet Transform (DWT) to break down time series into simpler components, which are then modeled using machine learning algorithms. Clustering-based and fuzzy logic-based models aim to uncover hidden structures or manage uncertainty in data. Moreover, optimization-enhanced hybrids employ algorithms like Particle Swarm Optimization (PSO) or Genetic Algorithms (GA) to fine-tune parameters across various modeling stages.

A particularly notable subcategory is that of sequential hybrids, where techniques are applied in consecutive stages.  exemplifies this approach, using EMD followed by machine learning to handle the nonlinear nature of precipitation data. Meanwhile, ensemble-based hybrids—such as bagging, boosting, or snapshot ensembles—combine multiple predictors to reduce variance and improve model stability. Finally, meta-model-based hybrids, like stacking, integrate the outputs of several base models through a higher-level meta-model. This is demonstrated by , where stacking significantly improved monthly rainfall prediction accuracy.

## 3.9 Postprocessing-based Hybrid Models

Researchers typically apply postprocessing as a set of statistical or probabilistic techniques after the primary model generates its predictions. These methods aim to correct systematic biases, calibrate output distributions, or enhance the interpretability of forecasts. Common independent techniques such as bias correction, Quantile Mapping (QM), and the Probability Integral Transformation (PIT) are widely used in climatology and hydrology to refine model outputs, especially in ensemble systems or physical models that often produce distributional mismatches.

In contrast, hybrid models based on postprocessing integrate these techniques directly into a two-stage predictive architecture. Rather than applying corrections after model execution, these hybrids embed postprocessing modules such as quantile regression forests, copula-based recalibrators, or neural calibration layers biases during training rather than as an isolated step.

For example, when researchers integrate Quantile Regression Forests (QRF), into a hybrid modeling pipeline, they achieve substantial improvements in daily precipitation forecast by addressing underdispersion and distributional skewness issues that standard machine learning models often neglect . Likewise, Ensemble Copula Coupling (ECC), which reshapes forecast ensembles while maintaining dependencies between variables, significantly improves uncertainly quantification in high impact weather events .

Another hybrid approach combines neural networks with embedded probabilistic calibration layers, allowing postprocessing to be learned jointly with the main forecasting model. These neural hybrid systems outperform traditional statistical postprocessing methods in terms of both sharpness and reliability of probabilistic forecasts (Rasp & Lerch, n.d.). Furthermore, comprehensive reviews have emphasized that integrating postprocessing directly within ensemble or ML architectures yields forecasts that are more adaptive to regime shifts, temporal nonstationarity, and nonlinear dependencias .

Classification of postprocessing-based hybrid models, which incorporate additional techniques following the initial prediction stage to enhance accuracy, correct systematic errors, or integrate multiple outputs as indicated in (Figure 6). This category is divided into three primary approaches:

1. **Statistical correction**, which includes methods such as bias correction and Bayesian Joint Probability (BJP). These techniques systematically adjust prediction errors to improve the reliability and accuracy of model outputs.

1. **Probabilistic model fusion**, represented by techniques like Bayesian Model Averaging (BMA) and Probabilistic Model Averaging, seeks to combine outputs from multiple models based on their probability distributions, producing more robust and uncertainty-aware predictions. For instance, the study *“**”* implemented probabilistic model fusion using BMA to enhance the accuracy and stability of rainfall predictions.

1. **Averaging-based postprocessing**, which encompasses methods such as prediction averaging, feature averaging, model averaging, and prediction selection. These approaches aggregate results from multiple models or configurations to improve generalization, reduce overfitting, and enhance model robustness. A clear example is presented in  who employed prediction averaging as a postprocessing strategy to refine model outputs.

![](Review_20250912_PerezManuel_assets/image_11.png)

Figure 6 | Structure and categorization of postprocessing based hybrid models.

## ---

1. Metrics for performance evaluation models

Performance evaluation is based on comparing quality metrics among the models used in the analyzed studies (Figure 7). These metrics assess the accuracy and reliability of model predictions by comparing the observed or actual values with the predicted results. This evaluation is conducted at various stages of predictive model development, including preparation, validation, and testing processes. Overall, these quality metrics provide essential insights into the model's effectiveness.

![](Review_20250912_PerezManuel_assets/image_03.png)

Figure 7 | Frequency of use distribution of performance metrics used in hybrid model evaluation.

Figure 7 shows the analysis of performance metrics, which reveals a clear preference for traditional error-based indicators in the assessment of monthly precipitation forecasting models. Root Mean Square Error (RMSE) is the most frequently utilized metric, appearing in 17 studies, followed by Mean Absolute Error (MAE) with 13 occurrences and Nash–Sutcliffe Efficiency (NSE) in 10 studies. These metrics dominate due to their intuitive interpretation and their sensitivity to deviations between observed and predicted values.

Deterministic correlation-based measures such as the coefficient of determination (R²) and Pearson’s correlation coefficient (R) are used less frequently, with 8 and 5 appearances respectively. In contrast, advanced or normalized metrics like Kling-Gupta Efficiency (KGE), Nash–Sutcliffe Coefficient (NSC), and relative indicators such as MAPE, RAE, or MARE, are used sporadically, suggesting a lower level of adoption in the literature. Metrics such as Bias Ratio (RBIAS), Volume Agreement Fraction (VAF), and Overall Index (OI) are rarely reported, each appearing in only one study. This pattern highlights the community's reliance on error magnitude metrics over normalized or distribution-sensitive measures, potentially limiting comparative analyses across heterogeneous climatic regions. Future studies would benefit from incorporating a broader set of performance indicators to enhance the robustness and comparability of hybrid model evaluations across diverse geographical and climatic contexts.

## 4.1 Root Mean Square Error (RMSE)

The metrics measures the average magnitude of the squared errors between predicted and observed values. It heavily penalizes large errors.

## (10)

## 4.2 Mean Absolute Error (MAE)

MAE calculates the average of the absolute differences between predicted and observed values. It provides a more balanced view of model errors.

## (11)

## 4.3 Coefficient of Determination (R²)

R² measures the proportion of variance in the observed data that is explained by the model.

## (12)

## 4.4 Nash–Sutcliffe Efficiency (NSE)

NSE evaluates the predictive skill of a model by comparing it to the performance of the mean of the observed data. Values close to 1 indicate high accuracy.

## (23)

## 4.5 Pearson Correlation Coefficient (r)

This metrics  measures the strength and direction of the linear relationship between predicted and observed values.

## (14)

## ---

1. REVIEW OF THE HYBRID MONTHLY PRECIPITATION PREDICTION MODELS

Hybrid models have gained increasing attention in recent years due to their ability to capture the complex, nonlinear nature of monthly precipitation patterns by combining statistical, machine learning, and signal decomposition methods. To illustrate this trend, Figure 8 shows the annual distribution of publications on hybrid monthly precipitation forecasting from 1996 to 2025, across three major databases: ScienceDirect, Scopus, and IEEE Xplore.

The results highlight a clear increase in research activity since 2017, with a significant surge from 2020 onward, peaking in 2023 with 14 publications. IEEE has become the most prominent source in recent years, reflecting a growing emphasis on AI-driven approaches. This trend confirms the rising relevance and scientific interest in hybrid forecasting models for climate and water resource applications.

![](Review_20250912_PerezManuel_assets/image_14.png)

Figure 8 | Trends in research on hybrid monthly precipitation prediction models. (The result for the year 2025 is partial; that is why its value is lower**).**

Table 2 | Main metrics used to evaluate the accuracy of hybrid models for monthly precipitation prediction.

| Ref | Study Zone | Resolución Temporal | Spatial Resolution | Variables | Hybrid Type | Based Algorithm |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Yangtze River Delta, on the southeast coast of China | Monthly | 1°x 1° | Number station, Name Station, Latitude, Longitude, Altitude, Mean Monthly Precipitation, Maximun Monthly Precipitation, Coefficient of Variation of Monthly Precipitation, Southern Oscillation Index, Western Pacific Subtropical High Intensity, See Level Pressure, Maximum Monthly Air Temperature, Minimum Monthly Air Temperature, Mean Monthly Pressure, Mean Vapor Pressure, Relative Humidity, Sunshine Duration. | Meta model based hybrids | KNN, XGB, SVR, ANN |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| Region of southwest Asia | Monthly | 1°x 1° | Mean Monthly Precipitation, Latitude, Longitude. | Meta model based hybrids | CanCM4i, COLA-RSMAS-CCSM4, GEM-NEMO, NASA-GEOSS2S |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (L. Tao et al., 2021) | Yangtze River basin (YRB) | Monthly | Regional (129 stations) | Mean Monthly Precipitation, Atlantic Multidecadal Oscilation, Sea Surface Temperature, Indian Ocean Dipole, Ocean Niño Index, Pacific Decadal Oscillation, Transition Niño Index, Western Hemisphere Warm Pool, Atlantic Meridional Mode, Arctic Oscillation, North Atlantic Oscillation, Eastern Atlantic/ Western Russia, North Pacific Pattern, Pacific North American Index, Quasi-Bienn North Atlantic Oscillation, Southern Os Eastern Atlantic/Western Russia, Western Pac North Pacific Pattern, Biavariate EN: Pacific North American Index, Multivariate ENSO Index, Global Average Temperature Anomalies, Solar Flux. | Sequential multiscale hybrid | LSTM, MLR |  |
|  |  |  |  |  |  |  |  |
| (Bojang et al., 2020) | Deji Reservoir Basins in Taiwan | Monthly | Regional (22 stations) | Mean Monthly Precipitation, | Sequential multiscale hybrid | LS-SVR, RF |  |
|  | Shihmen Reservoir Basins in Taiwan | Monthly |  |  | Sequential multiscale hybrid |  |  |
| (Luo et al., 2022) | Kunming Station, China | Monthly | Point (1 station) | Precipitation Products, Atmospheric Pressure, Sea Surface Temperature. | Sequential Hybrid | ARIMA, SVR, LSTM |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  | Hybrid decomposition-based ensemble |  |  |
|  |  |  |  |  | Sequential Hybrid |  |  |
| (Shen & Ban, 2023) | Lanzhou City, China | Monthly | Point (1 station) | Monthly Precipitation. | Descomposition based hybrid | SVM, LSTM, XGB, ARIMA |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Z. Zhou et al., 2021) | Eastern China | Monthly | Regional (25 stations) | Mean Wind Speed, Maximum Monthly Precipitation, Minimum Monthly Temprerature, Mean Monthly Pressure, El Niño Southern Oscillation, North Atlantic Oscillation. | Stacking | RF, ARIMA,ANN SVR, RNN |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Zandi et al., 2022) | Northwest Iran | Monthly | 0,25°x0,25° | Cloud Cover Data, Relative Humidity, Wind Speed and Direction | Meta model based hybrids | SVM, RF, MLP |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Kumar et al., 2021) | India | Monthly | Regional (36 stations) | Precipitation Products, Atmospheric Pressure, Sea Surface Temperature. | Optimization enhanced hybrid | DNN, ELM, WDNN, BLR |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| China | Monthly | Point (6 stations) | Precipitation Products, Atmospheric Pressure, North Atlatic Oscillation. | Sequential hybrid | CEEMD, WT |  |  |
|  |  |  |  |  |  |  |  |
| (X. Zhang & Wu, 2023) | Zhengzhou City | Monthly | 0,25°x0,25° | Precipitation Products, Atmospheric Pressure, Sea Surface Temperature, North Atlatic Oscillation. | Sequential multiscale hybrid with optimization | ELM, EMD-HHT, FFOA, CEEMD |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Tang et al., 2022) | Danjiangkou River Basin | Monthly | 0,25°x0,25° | Precipitation Products, Sea Surface Temperature, Atmospheric Pressure, El Niño Southern Oscillation, North Atlatic Oscillation, Standardized Precipitation Index. | Sequential Hybrid | RF, XGB, RNN, LSTM |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Yeditha et al., 2023) | Krishna River Basin | Monthly | 0,25°x0,25° | Atmospheric Pressure, Maximum Monthly Precipitation, Mean Monthly Precipitation. | Sequential multiscale hybrid | ELM, FFBN-NN, MLR |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Papacharalampous et al., 2023) | EE. UU | Monthly | 0,25°x0,25° | Altitude. | Meta model based hybrids | RF, XGB, LR, BRNN, MARS, GBM |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Parviz et al., 2021) | Tabríz, Rasht, Iran | Monthly | Point (2 stations) | Maximum Monthly Precipitation, Indian Ocean Dipole, El Niño Southern Oscillation, North Atlantic Oscillation, Sea Surface Temperature. | Stacking | SVR, GEP, GMDH, SARIMA |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Esmaeili et al., 2021) | Ardabil City | Monthly | Point (1 station) | Altitude, Maximum Monthly Precipitation, Mean Monthly Precipitation. | Sequential hybrid | ELM, WT |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Y. Li et al., 2021) | China | Monthly | 0,5°x0,5° | Maximum Monthly Precipitation, Mean Monthly Precipitation, Relative Humidity, Wind Speed and Direction, El Niño Southern Oscillation, North Atlantic Oscillation. | Probabilistic Model Averaging | LGB, XGB |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (X. Zhang et al., 2022) | Zhongwei City, China | Monthly | Point (2 stations) | Altitude, Maximum Monthly Precipitation, Mean Monthly Precipitation, Pacific Decadal Oscillation, Southern Oscillation Index. | Probabilistic Model Averaging | KNN, RF, SVR, ANN, LSTM |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Zerouali et al., 2023) | Sebaou River Basin | Monthly | Point (3 stations) | Mean Monthly Precipitation, Maximum Monthly Precipitation, Wind Speed and Direction, Relative Humidity, Sea Level Pressure. | Sequential hybrid | MLP, ELM |  |
|  |  |  |  |  |  |  |  |
| (Priestly et al., 2023) | Idukki, Kerala, India | Monthly | Point (Geographic coordinates) | Precipítation Products, Atmospheric Pressure, Sea Surface Temperature, Standardized Precipitation Index. | Optimization enhanced hybrid | SVM, ANN, KNN, MLR, GP |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Ridwan et al., 2021) | Tasik Kenyir, Terengganu, Malaysia | Monthly | Point (10 stations) | Mean Monthly Precipitation, Maximum Monthly Precipitation, Wind Speed and Direction, Relative Humidity, Sea Level Pressure, Relative Humidity. | Probabilistic Model Averaging | RNN, BDTR, DFR, BLR |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Chhetri et al., 2020) | Simtokha, Thimphu, Bhutan | Monthly | Point (1 station) | Relative Humidity, Wind Speed and Direction, Sea Level Pressure, Mean Monthly Precipitation, Meridional Wind, Altitude, El Niño Southern Oscillation. | Probabilistic Model Averaging | MLP,CNN, LSTM, LR, GRU, BLSTM |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Coşkun & Citakoglu, 2023) | Sakarya, Turkiye | Monthly | Point (1 station) | Altitude, Maximum Monthly Precipitation, North Atlantic Oscillation, Sea Level Pressure, Sea Surface Temperature, Southern Oscillation Index, Wind Speed and Direction. | Descomposition based hybrid | LSTM, ELM, EMD |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Ehteram et al., 2024) | Kashan Plain, Irán | Monthly | Point (5 Stations) | Longitude, Mean Monthly Precipitation, Maximum Monthly Precipitation, Relative Humidity, Atmospheric Pressure, Latitude, Satellite Derived Rainfall Estimates, Sea Surface Temperature | Prediction Averaging (FR), Feature Averaging (DRSN-TCN-RF) | RF, TCN, DRSN, OPA |  |
|  |  |  |  |  | Prediction Averaging (FR), Feature Averaging (DRSN-TCN-RF) |  |  |
|  |  |  |  |  | Descomposition based hybrid |  |  |
|  |  |  |  |  |  |  |  |
| (Ahmadi et al., 2024) | Sindh River basin, India | Monthly | Pointl (8 stations) | Precipitation Products, Standard Deviation, Coefficient of Variation, Approximate Coefficient (Level 3) | Descomposition based hybrids | RF, Kstar, GPR |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Parviz, 2020) | Rasht, Gorgan, Irán | Monthly | Point (2 stations) | Altitude, Atmospheric Pressure, Latitude, Longitude, Mean Monthly Precipitation, Solar Flux, Seasonal Cycle, Streamflow, Altitude, Atmospheric Pressure. | Sequential hybrid | SARIMA, ANN, SVM |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Guo et al., 2024) | Jinan, China | Monthly | Point (1 station) | Mean Monthly Precipitation, Maximum Monthly Precipitation, Relative Humidity, Maximum Monthly Temperature, Minimum Monthly Temperature. | Sequential hybrid | ANN, RNN, LSTM, CNN |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Hou et al., 2024) | Lanzhou, Gansu, China | Monthly | Point (1 station) | Atmospheric Pressure, Relative Humidity, Wind Spedd and Direction, Mean Wind Speed. | Descomposition based hybrid with optimization | LSTM, CPO-LSTM |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (X. Wu et al., 2021) | Jilin, China | Monthly | Point (3 stations) | Altitude, Longitude, Latitude. | Descomposition based hybrid | ARIMA. LSTM, GM, DGM, GM |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (H. Zhang et al., 2020) | Hubei, China | Monthly | Point (27 stations) | Mean Monthly Precipitation, Altitude, Relative Humidity, Latitude, Terrain Slope, Normalizaed Diference Vegetation Index, Daytima Surface Temperature. | Sequential hybrid | BP-ANN, MLR, RF, SVR, GPR |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| (Latif et al., 2024) | Sulaymaniyah, Iraq | Monthly | Point (1 station) | Mean Monthly Precipitation, Time Lags. | Sequential hybrid | ANN, SARIMA |  |
|  |  |  |  |  |  |  |  |
| (Katipoğlu & Keblouti, 2024) | El Kerma, Algeria | Monthly | Point (1 station) | Maximum Monthly Precipitation. | Descomposition based hybrid | SVM, ANN, GPR |  |

Table 3 | Comparative Performance of Hybrid and Ensemble Models for Monthly Precipitation Prediction: Evaluation Metrics and Percent Performance Improvement

| Ref | Based Algorithm | Bets Model | Evaluation Metrics | % Performance |  |
| --- | --- | --- | --- | --- | --- |
| KNN, XGB, SVR, ANN | ANN, Stacking | ANN: MAE=42,47; R^2= 0,532; RMSE= 60,92; Stacking; MAE=41,65; R^2= 0,526; RMSE= 61,51 | 0,96 |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| CanCM4i, COLA-RSMAS-CCSM4, GEM-NEMO, NASA-GEOSS2S | RF, ANN | RF: KGE= 0,68 ; NSE= 0,6; r= 0,77; RMSE= 24,83; ANN: KGE= 0,64 NSE= 0,52; r= 0,74; RMSE= 26,08 | 4,21 |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (L. Tao et al., 2021) | LSTM, MLR | MLSTM-AM, LSTM | MLSTM-AM: NSE=0,46; RAE= 0,41; LSTM;NSE= 0,44; RAE= 0,44 | 6,81 |  |
|  |  |  |  |  |  |
| (Bojang et al., 2020) | LS-SVR, RF | SSA-LSSVR , SSA-RF | SSA-LSSVR: NSE= 0,86; RMSE= 75,29; SSA-RF: NSE= 0,63; RMSE= 121,76 | 38,16 |  |
|  |  | SSA-RF, SSA-LSSVR | SSA-RF: NSE= 0,82; RMSE= 98,75; SSA-LSSVR: NSE= 0,67; RMSE= 132,81 | 25,64 |  |
| (Luo et al., 2022) | ARIMA, SVR, LSTM | EEMD-BMA, EEMD-LSTM | EEMD.BMA MSE= 298,6359; CP=93,75; RMSE=17,28; MAE=12,7; R^2=0,9573 MW=60,315 EEMD-LSTM: MSE=301,94; RMSE= 17,37; MAE= 12,77; R^2= 0,956 | 0,5 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Shen & Ban, 2023) | SVM, LSTM, XGB, ARIMA | CEEMDAN-SVM-LSTM, CEEDMAN-LSTM | CEEMDAN-SVM-LSTM: RMSE=7,68; MSE=58,98; MAE=5,58; R^2=0,95; CEEDMAN-LSTM: RMSE=18,09; MSE= 327,25; MAE=12,43; R^2=0,71 | 57,54 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Z. Zhou et al., 2021) | RF, ARIMA,ANN SVR, RNN | RF, ANN | RF: MAE=25,7; RMSE=40,8; R^2=0,82; ANN: MAE=26,9; RMSE=41,7; R^2=0,82 | 2,16 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Zandi et al., 2022) | SVM, RF, MLP | Stck-Las-Rsc, Stck-Las | Stck-Las-Rsc: RMSE=25,4; MAE=15,1; RBIAS= -1,4; KGE=0,88; Stck-Las: RMSE=25,7; MAE=15,2; RBIAS= -1,6; KGE=0,89 | 1,16 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| DNN, ELM, WDNN, BLR | WDNN, WBBO-ELM | WDNN: RMSE=185,9; NSE=0,96; MAE=130,86; R^2= 0,98; WBBO-ELM: RMSE=194,4; NSE=0,957; MAE=132,47; R^2= 0,979 | 4,37 |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  | CEEMD, WT | TVF-EMD-ENN, WT-ENN | TVF-EMD-ENN: RMSE=23,775; MAE=18,474; NSE= 0,97; WT-ENN: RMSE=35,5; MAE=28; NSE= 0,95 | 33,03 |  |
|  |  |  |  |  |  |
| (X. Zhang & Wu, 2023) | ELM, EMD-HHT, FFOA, CEEMD | CEEMD-ELM-FFOA, CEEMD-ELM | CEEMD-ELM-FFOA; MAE=0,55; RMSE=0,81; MAPE=1,39; CEEDM-ELM: MAE=0,63; RMSE=0,92; MAPE=3,23 | 11,96 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Tang et al., 2022) | RF, XGB, RNN, LSTM | SMOTE-km-XGB, SMOTE-km-RF | SMOTE-km-XGB: ACC=05; Pg score=80; SMOTE-km-RF: ACC=05; Pg score=70 | 12,5 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Yeditha et al., 2023) | ELM, FFBN-NN, MLR | WT-ELM, WT-FFBP-NN | WT-ELM: RMSE=0,03; MAE=0,052; r=0,925; NSE=0,855; WT-FFBP-NN: RMSE=0,033; MAE=0,055; r=0,892; NSE=0,796 | 9,1 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Papacharalampous et al., 2023) | RF, XGB, LR, BRNN, MARS, GBM | LR, polyMARS | LR: MSE= 39,77; ployMARS: MSE= 39,18 | 1,48 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Parviz et al., 2021) | SVR, GEP, GMDH, SARIMA | GA, SARIMA-SVR | GA: NSE=0,97; RMSE=11,48; RRMSE=0,1; MSE=131,84; RPD=6,57; APB=0,078; Dm=0,99; U2=0,088;AMPE=4,52; SARIMA-SVR: NSE=0,92; RMSE=20,02; RRMSE=0,187; MSE=400,9; RPD=3,77; APB=0,142; Dm=0,97; U2=0,15; AMPE= 4,58 | 42,66 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Esmaeili et al., 2021) | ELM, WT | WORELM 14, WORELM 6 | WORELM 14: MARE=4,345 ; r=0,965; NSC=0,965; VAF=93,16; WORELM 6: MARE=6,5; r=0,92; NSC=0,90; VAF=92,5 | 33,15 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Y. Li et al., 2021) | LGB, XGB | SEAS4 & SEAS5 | SEAS 5: MAE=0,35; PCC=0,9; SEAS 4: MAE=0,65; PCC=0,8; | 11,11 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (X. Zhang et al., 2022) | KNN, RF, SVR, ANN, LSTM | CEEMD-FCMSE-Stacking, CEEDM-LSTM | CEEMD-FCMSE-Stacking: MAE=4,26; RMSE=7,71; R^2=0,923; CEEDM-LSTM: MAE= 7,62; RMSE= 10,16; R^2=0,88 | 24,11 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Zerouali et al., 2023) | MLP, ELM | Bat-ELM, MLP-PSO | Bat-ELM: RMSE=11,89; NSE=0,985; r=0,993; R^2=0,986; MLP-PSO: RMSE=30,50; NSE=0,879; r=0,951; R^2=0,905; | 61,31 |  |
|  |  |  |  |  |  |
| (Priestly et al., 2023) | SVM, ANN, KNN, MLR, GP | LSO-ABR Model 3, LSO-ABR Model 1 | LSO-ABR Model 3: RMSE=52,39; MAE=39,89; LSO-ABR Model 1: RMSE=53,29; MAE=40,33 | 1,69 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Ridwan et al., 2021) | RNN, BDTR, DFR, BLR | BDTR, DFR | BDTR; MAE=0,0123; RMSE=0,0335; R^2=0,999 DFR: MAE=0,0946; RMSE=0,1565; R^2=0,7623 | 78,84 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Chhetri et al., 2020) | MLP, CNN, LSTM, LR, GRU, BLSTM | BLSTM-GRU, LSTM | BLSTM-GRU: RMSE=0,087; MSE=0,0075; R^2=0,87; PCC= 0,938; LSTM: MSE=0,0075; R^2=0,87; PCC=0,90 | 4,05 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Coşkun & Citakoglu, 2023) | LSTM, ELM, EMD | LSTM & EMD-ELM | LSTM: MAE=0,11; NSE=0,97; R^2=0,97; RMSE=0,17; OI=0,97; EMD-ELM: MAE=0,22; NSE=0,95; R^2=0,96; RMSE=0,25 OI=0,95 | 32 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Ehteram et al., 2024) | RF, TCN, DRSN, OPA | DTR, TCN-RF | DTR: RMSE=0,15; MAE=0,195; NSE=0,96; R^2=0,992; TCN-RF: RMSE=0,19; MAE=0,195; NSE=0,94; R^2=0,985 | 21,05 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Ahmadi et al., 2024) | RF, Kstar, GPR | Wavelet-RF, Wavelet-Kstar | Wavelet-RF: RMSE=34,47; MAE=20,53; KGE=0,917; WI=0,884; Wavelet-Kstar: RMSE=36,59; MAE=20,339; KGE=0,878; WI=0,871 | 5,79 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Parviz, 2020) | SARIMA, ANN, SVM | SARIMA-SVM, ANN | SARIMA-SVM: RMSE=14,18; MAE=9,55; RRMSE=0,39; d= 0,84 ; U1=0,16; ANN: RMSE=23,88; MAE= 18,93; RRMSE=0,66; d= 0,43; U1=0,26 | 40,62 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Guo et al., 2024) | ANN, RNN, LSTM, CNN | CNN-LSTM, CNN | CNN-LSTM: RMSE=0,6292; MAE=0,5048; r=0,998; CNN: RMSE=0,8015; MAE=0,6680; r=0,996 | 21,5 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Hou et al., 2024) | LSTM, CPO-LSTM | VMD-CPO-LSTM, VDM-LSTM | VMD-CPO-LSTM: RMSE=15,52; MAE=9,98; R^2=0,9039; NSE=0,9039; VMD-LSTM: RMSE=22,02; MAE=13,74; R^2=0,806; NSE=0,806; | 29,52 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (X. Wu et al., 2021) | ARIMA. LSTM, GM, DGM, GM | W-AL (Wavelet-ARIMA-LSTM), LSTM | W-AL: RMSE=31,308; MAE=22,853; R^2=0,712 LSTM: RMSE=36,024; MAE=25,865; R^2=0,618 | 13,09 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (H. Zhang et al., 2020) | BP-ANN, MLR, RF, SVR, GPR | BP-ANN, Rattional Quadratic GPR | BP-ANN: RMSE=43,32; MAE=31,9; R^2=0,64; MSE=1876,5; Rattional Quadratic GPR: RMSE=51,22; MAE=33,13; R^2=0,59; MSE=2623,8 | 15,42 |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
|  |  |  |  |  |  |
| (Latif et al., 2024) | ANN, SARIMA | SARIMA-ANN, ANN Model 4 | SARIMA-ANN: R^2=0,98; RMSE= 11,5; ANN Model 4: R^2=0,43; RMSE= 51,002 | 77,45 |  |
|  |  |  |  |  |  |
| SVM, ANN, GPR | W-LSVM, LSVM | W-LSVM: R^2=0,78; RMSE=8,1; MAE=5,9; LSVM: R^2=0,78; RMSE=8,18; MAE=5,9 | 0,98 |  |  |

(RMSE) and (MAE) stand out among the six most used evaluation metrics. However, it is essential to note that (NSE), (R2), and (MSE) are also widely used alongside RMSE and MAE. other evaluation metrics include CC, RBIAS, ACC, OR CRPS. as shown in Figure 7, there is a wide variety of metrics, and choosing the most appropriate one for the model being evaluated is essential to ensure rigor, efficiency and accuracy. The most review articles compare simple models with hybrid or ensemble models (Table 2), showing superior performance of hybrid and ensemble models over simple models (Table 3 ). However, few studies compare hybrid models with other hybrid or ensemble models. In the models where such comparisons were made, it was observed that some models perform better for a specific geographic area. The forms of hybridization (Hybrid predictive models ) include Hybridization by Preprocessing (HP), Hybridization of Neural Networks with Optimization Algorithms (HRNAO), Hybridization of Machine Learning Models with Signal Decomposition (HMMLDS), and Hybridizations of Statistical Time Series Models with Neural Networks (HMESTRN). Hybrid and ensemble models present various variations and construction methods, adapting to different contexts and specific needs by leveraging the strengths of different techniques, methods, algorithms, and models.

Table 4 | Performance Comparison of Different Hybrid Model Types for Prediction: Average RMSE and MAE in Selected Studies

| Model Type | Studies | Average RMSE | Average MAE |
| --- | --- | --- | --- |
| Meta model-based hybrids | 4 | 49.94 | 33.01 |
| Sequential multiscale hybrid | 5 | 43.59 | 9.27 |
| Sequential hybrid | 9 | 15.71 | 16.57 |
| Decomposition based hybrid | 7 | 14.95 | 10.18 |
| Probabilistic Model Averaging | 5 | 0.89 | 0.71 |
| Optimization enhanced hybrid | 2 | 161.10 | 94.80 |
| Stacking | 2 | 11.48 | — |
| Prediction Averaging / Feature Averaging | 1 | 0.19 | 0.195 |
| Decomposition based hybrid with optimization | 1 | 15.52 | 9.98 |

## ---

## DISCUSSION

Hybrid and ensemble models have consistently demonstrated better performance compared to individual models, as shown in Table 2 and Table 3 . This finding reinforces the idea that integrating signal decomposition techniques with advanced machine learning algorithms significantly enhances predictive capability. The following subsections provide a more detailed comparative analysis, exploring specific techniques and highlighting key metrics to identify particularly robust methodologies. These findings are consistent with previous studies that have demonstrated the advantages of hybrid and ensemble approaches. This could be attributed to differences in the datasets used and the specific climatic conditions of each region studied. Most studies have focused on developing and evaluating simple machine learning models compared with hybrid models; however, there has been sufficient development and evaluation among hybrid or ensemble models themselves. To continue improving prediction accuracy, it is advisable to investigate new combinations of signal decomposition techniques and machine learning algorithms. It is important to develop and evaluate hybrid models with small variations, whether in the signal decomposition stage, training, result combination, or other aspects, to determine which combinations may prove to be more beneficial. Most of the reviewed studies were conducted in mountainous regions or areas combining mountains and plains, featuring diverse terrains such as mountain plateaus, river valleys, and coastal plains. Elevations ranged widely from -431 meters below sea level to a maximum of 8,848 meters, reflecting a broad geographical scope. Eastern and northwestern China, as well as mountainous regions of Iran, were the most studied locations. Regarding data sources, the most frequently used variables included mean monthly precipitation, obtained from both local meteorological stations and satellite products. Some studies also incorporated topographic variables like altitude and atmospheric pressure, alongside global climate phenomena such as ENSO.

## 6.1 Systematic Comparison of Specific Hybrid Techniques

A comprehensive systematic analysis (Table 2) reveals specific hybrid technique combinations consistently demonstrating superior performance (Table 3 ). For instance, standard normalization combined with ensemble techniques such as stacking and hyperparameter optimization (Gu et al., 2022; Pakdaman et al., 2022) significantly enhances predictive accuracy, particularly reflected in metrics like RMSE (up to 20% reduction) and R² (30-40% improvement). Techniques involving prior transformations such as (SSA) combined with (SVM) have also shown remarkable effectiveness in highly seasonal and variable contexts, highlighting the critical importance of selecting specific methodological combinations tailored to local climatic variability (Bojang et al., 2020; Shen & Ban, 2023).

## 6.2 Detailed comparative analysis of model performance by specific metrics

A detailed quantitative analysis based on diverse metrics (Table 2) reveals RMSE and MAE as predominant metrics utilized in hybrid model evaluation (Table 3 ). Hybrid models employing optimization algorithms such as Genetic Algorithms and Particle Swarm Optimization consistently yield notably lower RMSE and MAE values compared to simpler and less optimized hybrid techniques. Furthermore, metrics like (NSE) and the Coefficient of Determination (R²) have proven essential in evaluating model alignment with observed real-world behaviors, particularly when hybrid methods include signal decomposition using Wavelet or CEEMDAN techniques (Ahmadi et al., 2024; Zhang & Wu, 2023). To visually illustrate these findings, the following graphics (Figure 9) clearly compare hybrid techniques according to key metrics, facilitating rapid interpretation and strengthening the analytical presentation.

![](Review_20250912_PerezManuel_assets/image_07.png)Figure 9 | Comparative performance of hybrid models according to RMSE and MAE metrics.

The following are examples that demonstrate the performance of hybrid models. In  the use of hybrid approaches leads to an improvement of approximately 46%, primarily due to a significant reduction in RMSE. In the article published be , there is an improvement of approximately 12% to 20% is observed in the R² metric. These studies incorporate decomposition techniques (such as wavelets) and optimization methods (metaheuristics), which yield better results, particularly for datasets characterized by high nonlinearity and seasonality. In , the RMSE is reduced by about 64.3%, indicating a much more accurate prediction. The R² value increases from 0.60 to 0.95, reflecting a substantial improvement in the model's ability to fit the actual data. This study demonstrates how the implementation of CEEMDAN decomposition enables the simplification of signals, enhancing the predictive capabilities of LSTM and SVM when used in combination. Another relevant case is presented in , which analyzes a simple model (ENN). This model shows lower accuracy, with reduced NSE values and higher RMSE and MAE. When hybrid models incorporating signal decomposition techniques such as TVF-EMD, WT, and CEEMD are applied, performance improves significantly. The TVF-EMD-ENN model achieved the best overall results, with an improvement of approximately 14.3% in NSE and 5% to 10% reductions in RMSE and MAE, compared to the independent ENN model. Although the superiority of hybrid models over individual models is evident, there is a notable lack of comparative analyses among hybrid models themselves. The performance differences observed across various regions suggest that the specific choice of hybrid techniques should be carefully tailored to particular climatic and geographical characteristics. Future studies could explore this dimension in greater depth, providing clear guidelines for the optimal selection of models. Moreover, the limited use of advanced metrics such as the Kling-Gupta Efficiency (KGE) and other distribution-sensitive measures highlights an opportunity to enhance the robustness of comparative evaluations. Incorporating these metrics would facilitate more accurate comparisons, especially in regions with extreme climatic variability. In , the best-performing model was WDNN, achieving RMSE and MAE values of 185.9 mm and 130.86 mm, respectively. These results were further divided among the 36 monitoring stations to obtain more precise error estimates per station and study area, considering that the analysis was conducted at the national level; in this case, India. Figure 9 uses a logarithmic scale on both the RMSE and MAE axes to allow for meaningful visualization of error magnitudes across a wide range of hybrids models. By applying a log₁₀ transformation, the chart effectively compresses large numerical disparities, making both small and large values visible on the same scale. This approach enhances interpretability, especially when some models produce extremely low errors while others yield significantly higher values. The color gradient represents the relative magnitude of the error, with darker shades indicating higher RMSE or MAE. The left panel displays the average RMSE for each hybrid model, while the right panel presents the average MAE. Models such as SSA+LS-SVR and ANN exhibit the highest error magnitudes, while architectures like SEA64 & SEAS5, WT-ELM, and BLSTM-GRU consistently achieve the lowest errors. This dual-panel layout allows a direct visual comparison of performance consistency between RMSE and MAE, and highlights which hybrid strategies offer better generalization and predictive accuracy across multiple studies.

![](Review_20250912_PerezManuel_assets/image_04.png)

## **6.3 ****Limitations ****and**** ****S****ources of ****U****ncertainty in ML ****P****recipitation ****P****rediction**

Key limitations include: (i) historical dependence and non-stationarity; relationships can drift under climate variability . (ii) Initial state sensitivity; even at monthly leads, conditioning on antecedent states matters. (iii) Sparse/biased observations; gauge representativeness and satellite retrieval uncertainties. (iv) Shortcut learning and overfitting, especially with small samples. (v) Data leakage; e.g., random cross-validation (CV) that mixes spatial neighbors or months. (vi) Uncertainty quantification gaps; deterministic hybrids dominate despite the need for calibrated distributions. Mitigations: spatial/temporal cross-validation (leave one region/month out), hindcast evaluation, leakage checks, feature ablation, and reporting calibrated uncertainty (quantile methods/QRF, EMOS, ECC). Operational cautions raised for ML forecasts , and recommend CV designs tailored to spatial data .

## **6.****4**** ****Complementarity of ****P****hysical ****Numerical Weather Prediction (****NWP****)**** and**** ****ML**

ML does not replace NWP; it complements it at three stages: (1) Data assimilation – learned priors and emulators can accelerate or correct variational/ensemble methods . (2) Downscaling and bias correction/post processing – hybrids (quantile mapping, QRF, EMOS, ECC, neural calibration) correct distributional mismatches and refine spatial detail . (3) Surrogates – emulate expensive components for scenario exploration . Monthly hybrid predictors reviewed here primarily operate in (2) and occasionally (3). Recent ML augmented global forecasting  and generative ensemble emulation  illustrate progress but also reinforce the need for rigorous calibration and verification.

## **6.****5**** ****Promising ****A****reas and ****U****nderexplored ****H****ybrid ****T****echniques**

Despite the increasing popularity of hybrid techniques, certain specific methodologies remain significantly underexplored. Particularly, postprocessing techniques based on Probabilistic Model Averaging (PMA), advanced hybrid techniques integrating neural networks with bio-inspired optimization methods (Honey Badger Algorithm, Bat Algorithm), and sequential preprocessing techniques such as SMOTE-K-means present clear opportunities for future development. These approaches exhibit significant potential, particularly in geographically and climatically complex contexts, highlighting an urgent need for further specific studies evaluating their comparative effectiveness in diverse operational scenarios. In the study conducted by  in the Boyacá department of Colombia which was carried out in a predominantly mountainous region, characterized by significant altitudinal variability and complex orographic conditions. A total of 757 geographic points were analyzed using CHIRPS 2.0 satellite data at a spatial resolution of 0.05°, but no clustering techniques were explicitly applied to group areas by elevation. However, the integration of elevation-based clustering methods, such as k-means or DBSCAN, could greatly enhance model performance by segmenting the region into homogeneous subzones with similar rainfall dynamics, thus improving the models’ ability to capture localized patterns. The study compared three predictive models: ARIMA, Random Forest Regressor, and (LSTM) neural network, with the LSTM model achieving the best performance (RMSE = 19.43 mm, MAE = 10.39 mm, R² = 0.92). This model was used to forecast monthly precipitation over a 16-month period (September 2023 to December 2024), demonstrating a high capacity to predict seasonal variations and extreme events, including rainfall levels exceeding 700 mm in municipalities like Cubará. These findings emphasize the value of incorporating local topographic features into modeling processes and suggest that altitude-based clustering could further improve spatial prediction accuracy in mountainous regions.

## ---

## ABBREVIATIONS

This section provides a comprehensive list of abbreviations used throughout the article. Table 5  summarizes the key terms, including abbreviations related to predictive models, data preprocessing techniques, data postprocessing techniques and optimization parameters. This table aims to facilitate the reader´s understanding by clarifying the terminology used in the methodological and analytical discussions.

Table 5 | Abbreviations of Models, Techniques and Optimization Parameters.

| Abbrevation | Description | Abbrevation | Description |
| --- | --- | --- | --- |
| ACF | Autocorrelation Function | GP | Genetic Programming |
| ACF and CCF | Autocorrelation Function / Cross-Correlation Function | GPR | Gaussian Process Regression |
| ADF | Augmented Dickey-Fuller Test | GRU | Gated Recurrent Unit |
| ANFIS | Adaptive Neuro-Fuzzy Inference System | HBA | Honey Badger Algorithm |
| ANN | Artificial Neural Network | HYPERPARAMETER TUNING | The process of fine-tuning model parameters. |
| ARIMA | AutoRegressive Integrated Moving Average | IMFs | Intrinsic Mode Functions |
| ARIMA | AutoRegressive Integrated Moving Average | K MEANS | K-means Clustering |
| ARIMA-STL | Autoregressive integrated moving average descompistion with STL | KNN | K-Nearest Neighbors |
| BAT | Bat Algorithm | LGB | Light Gradient Boosting Machine |
| BBO | Biogeography-Based Optimization | LOMOCV | Leave-One-Month-Out Cross-Validation |
| BBO-ELM | Biogeography-based Extreme Learning Machine | LR | Linear Regression |
| BDTR | Boosted Decision Tree Regression | LSO | Leave-Some-Out |
| Bias Correction | A method to adjust systematic biases in prediction models | LS-SVR | Least Squares Support Vector Regression |
| BJP | Bayesian Joint Probability | LSTM | Long Short-Term Memory |
| BLR | Bayesian Linear Regression | LTP | Lead times processing |
| BLSTM | Bidirectional Long Short-Term Memory | MARS | Multivariate Adaptive Regression Splines |
| BLSTM-GRU | Bidirectional LSTM combined with GRU | MIN-MAX NORMALIZATION | Scales data to a range between 0 and 1 |
| BPNN | Backpropagation Neural Network | MLP | Multilayer Perceptron |
| BRFFNN | Bayesian regularized Feedforward Neural Network | MLR | Multiple Linear Regression |
| BRNN | Bidirectional Recurrent Neural Network | MLSTM | Multiscale LSTM |
| CanCM4i | Canadian Climate Model version 4 | MLSTM-AM | Multiscale LSTM with Attention Mechanism |
| CEEMD | Complete Ensemble Empirical Mode Decomposition | MODWT | Maximal Overlap Discrete Wavelet Transform |
| CEEMDAN | Complete Ensemble Empirical Mode Decomposition with Adaptive Noise | NASA-GEOSS2S | NASA Global Earth Observing System Subseasonal to Seasonal |
| CNN | Convolutional Neural Network | NMF | Non-negative Matrix Factorization |
| COLA-RSMAS-CCSM4 | Community Climate System Model by COLA-RSMAS | OPA | Optimization algorithm (unspecified) |
| CPO | The Crested Porcupine Optimization (CPO) | PACF | Partial Autocorrelation Function |
| CPO-LSTM | The crested porcupine optimization algorithm for LSTM | PCA | Principal component analysis (PCA) method |
| DFR | Decision Forest Regression | PI | Predictor Importance |
| DNN | Deep Neural Network | PSO | Particle Swarm Optimization: |
| DRSN | Deep Residual Shrinkage Network | PSO-ELM | Particle Swarm Optimization optimized ELM |
| DSABASRNN | Deep Self Attention-Based Augmented SRNN | R/S Analysis | Rescaled Range Analysis |
| DWT | Discrete Wavelet Transform | RF | Random Forest |
| EEMD | Ensemble Empirical Mode Decomposition | RNN | Recurrent Neural Network |
| EEMD | Ensemble Empirical Mode Decomposition | RT | Regression Trees |
| EEMD-ARIMA | EEMD combined with ARIMA | SARIMA | Seasonal ARIMA |
| EEMD-LSTM | EEMD combined with LSTM | SARIMA | Seasonal ARIMA: An extension of ARIMA that incorporates seasonal components into time series modeling. |
| EEMD-SVR | EEMD combined with Support Vector Regression | SCE | The Shuffled Complex Evolution (SCE) |
| ELM | Extreme Learning Machine | SHANNON ENTROPY | Is a measure of uncertainty or randomness in a dataset. |
| EMD | Empirical Mode Decomposition | SMOTE | Synthetic Minority Oversampling Technique |
| EMD | Empirical Mode Decomposition | SMOTE + KMEANS | Sequential Preprocessing combinations |
| EMD-ELM | EMD combined with ELM | SSA | Singular Spectrum Analysis |
| EMD-HHT | EMD using Hilbert–Huang Transform | SSA-LSSVR | Singular Spectrum Analysis combined with LSSVR |
| FBP | Facebook Prophet ModeL | SSA-RF | Singular Spectrum Analysis combined with Random Forest |
| FCMSE | Fuzzy Comprehensive Mean Square Error | SVM | Support Vector Machine |
| FFA | The Firefly Algorithm (FFA) | SVN/ANN | Support Vector Network / Artificial Neural Network |
| FFBP-NN | Feedforward Backpropagation Neural Network | SVR | Support Vector Regression |
| FFNN | Feedforward Neural Network | TCN | Temporal Convolutional Network |
| FFOA | Fruit Fly Optimization Algorithm | TPS | Thin Plate Spline |
| FFOA | The Fruit Fly Optimization Algorithm (FFOA) | TVF | Time-Varying Filtering |
| GA | Genetic Algorithm | TVF-EMD | Time-Varying Filtering with EMD |
| GBM | Gradient Boosting Machine | VMD | Variational Mode Decomposition |
| GCM SEAS4 y SEAS5 | Outputs from global climate models (GCMs) of the SEAS4/SEAS5 systems from ECMWF | WDNN | Weighted Deep Neural Network |
| GEM-NEMO | Global Environmental Multiscale model – NEMO | WELM | Weighted Extreme Learning Machine |
| GEP | Gene Expression Programming | WORELM | Weighted Online Random ELM |
| GMDH | Group Method of Data Handling | WT | Wavelet Transform |
| GM-OPA | Generalized Model with Optimal Predictor Algorithm | XGB | Extreme Gradient Boosting |

## **_________****____****_**

## CONCLUSIONS

Hybrid and ensemble machine learning approaches now dominate monthly precipitation research and are likely to remain the growth engine of the field. Our systematic synthesis shows that data decomposition hybrids (CEEMD-ELM-FFOA, TVF-EMD-ENN) and multimodal stacking ensembles outperform single models by up to one order of magnitude in RMSE, MAE and NSE particularly when forecasting extremes and long series. Bio-inspired optimizers (GA, PSO, FFA, LSO) further boost accuracy by streamlining feature selection and hyper-parameters for MLP, XGBoost or ABR models. Sequential hybrid models and probabilistic model averaging also demonstrated outstanding accuracy, with some studies reporting NSE values above 0.95 and R² up to 0.98. These findings confirm the substantial benefits of hybridization in improving prediction reliability, especially when combining signal decomposition techniques with neural networks or optimization algorithms.

Despite these gains, two gaps persist: (1) few studies pit hybrids directly against ensembles, and (2) probabilistic post-processing and uncertainty quantification remain under-explored. Robust cross-validation across climates and orography’s especially in complex mountainous terrain is therefore essential. Future work should, compare state-of-the-art hybrids and ensembles under identical data splits and advanced metrics (KGE, CRPSS), incorporate spatial temporal transfer test to assess generalizability beyond the original study area, embed explain ability layers to link learned patterns with physical drivers and foster operational trust, publish neural or negative results to curb performance inflation and map true research frontiers. Accurate, interpretable rainfall forecasts are already transforming crop planning, water-allocation and early-warning systems capabilities that will only grow in importance as climate variability intensifies.

## **_________****_________________**

## DATA AVAILABILITY STATEMENT

The data that support the findings of this study are available on request from the corresponding author, upon reasonable request.

## **_________****_________****_**

## CONFLICT OF INTEREST

The authors declare there is no conflict.

## ---

## REFERENCES
