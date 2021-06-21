# Health-Insurance
Section 1: Business Understanding
An insurance company that has provided Health Insurance Products to its customer wants to expand its business to Vehicle Insurance. It needs to build a model to predict whether customers from the past year are interested in its new Vehicle Insurance Products to plan for marketing strategies. First, we will analyze how some variables are related to customers' interests. Then, we will build a machine learning model to classify whether a customer is interested. Customers forecasted to be interested in this product will become the target market and receive ads promoting this product.
# Section 2: Data Understanding
![image](https://user-images.githubusercontent.com/77747784/122756395-fbc67b80-d2b3-11eb-8186-dafed366ec3b.png)
# Set id as index
df.set_index('id',inplace=True)
df.head()
![image](https://user-images.githubusercontent.com/77747784/122756730-6e375b80-d2b4-11eb-987d-cfc2dded14cc.png)
![image](https://user-images.githubusercontent.com/77747784/122756852-90c97480-d2b4-11eb-984b-dd3001453e06.png)
![image](https://user-images.githubusercontent.com/77747784/122756997-be162280-d2b4-11eb-99d2-45d14cd8488f.png)
![image](https://user-images.githubusercontent.com/77747784/122757058-cf5f2f00-d2b4-11eb-90e6-a82ecce3badf.png)
![image](https://user-images.githubusercontent.com/77747784/122757187-f9b0ec80-d2b4-11eb-924f-2beec7c49abf.png)
![image](https://user-images.githubusercontent.com/77747784/122757315-1baa6f00-d2b5-11eb-889e-4cbd3a181a67.png)
![image](https://user-images.githubusercontent.com/77747784/122757414-38df3d80-d2b5-11eb-9287-93b7f9144243.png)
There are noticeable differences in customers interested proportion based on Region_Code. Because this variable is categorical, we will use dummies to encode this variable in the next section.
![image](https://user-images.githubusercontent.com/77747784/122757504-557b7580-d2b5-11eb-8a00-4cb977ce3009.png)
![image](https://user-images.githubusercontent.com/77747784/122757599-6d52f980-d2b5-11eb-80d1-d6c7b945c0db.png)
![image](https://user-images.githubusercontent.com/77747784/122757651-7e036f80-d2b5-11eb-8611-55d56a815dce.png)
![image](https://user-images.githubusercontent.com/77747784/122757701-91aed600-d2b5-11eb-86c7-4c0422d901a8.png)
![image](https://user-images.githubusercontent.com/77747784/122757768-a5f2d300-d2b5-11eb-9308-e3d620616c1e.png)
![image](https://user-images.githubusercontent.com/77747784/122757861-be62ed80-d2b5-11eb-8dd1-ab5f707deb08.png)
![image](https://user-images.githubusercontent.com/77747784/122757930-d0dd2700-d2b5-11eb-852b-c8507163d459.png)
![image](https://user-images.githubusercontent.com/77747784/122757983-e18d9d00-d2b5-11eb-94a3-7317b311d1c1.png)
![image](https://user-images.githubusercontent.com/77747784/122758024-ef432280-d2b5-11eb-83ce-91fd41dede29.png)
![image](https://user-images.githubusercontent.com/77747784/122758070-fd913e80-d2b5-11eb-8a27-7c2c661de4d6.png)
![image](https://user-images.githubusercontent.com/77747784/122758140-14d02c00-d2b6-11eb-9ef6-22c39792b01e.png)
![image](https://user-images.githubusercontent.com/77747784/122758230-2e717380-d2b6-11eb-93b0-ff7abde75264.png)
![image](https://user-images.githubusercontent.com/77747784/122758290-3f21e980-d2b6-11eb-9f7a-49f8c0fcca57.png)
![image](https://user-images.githubusercontent.com/77747784/122758345-506af600-d2b6-11eb-89b7-32fb0f94ecc9.png)
![image](https://user-images.githubusercontent.com/77747784/122758408-64aef300-d2b6-11eb-91cb-86578f4afe16.png)
![image](https://user-images.githubusercontent.com/77747784/122758485-78f2f000-d2b6-11eb-9a89-86c78231e121.png)
![image](https://user-images.githubusercontent.com/77747784/122758542-8c05c000-d2b6-11eb-9ed4-c282edca3f4f.png)
![image](https://user-images.githubusercontent.com/77747784/122758605-9f189000-d2b6-11eb-8037-d8b935e29f47.png)
![image](https://user-images.githubusercontent.com/77747784/122758707-bce5f500-d2b6-11eb-914b-f95146d5ab54.png)
![image](https://user-images.githubusercontent.com/77747784/122758777-d1c28880-d2b6-11eb-8440-e997e153b025.png)
With 0.8 AUC-ROC score, the logistic regression model is good enough. This model is also exceptional for predicting interested customers as shown by the 94% recall (target market will be wide enough to cover most of the interested customers). Although the precision is only 28% (28% of advertised customers is interested), it is acceptable because advertising to random customers will only get 12% precision (the proportion of interested customers in the dataset is approximately 12%).
![image](https://user-images.githubusercontent.com/77747784/122758875-edc62a00-d2b6-11eb-9969-aba086f803d5.png)
As analyzed in the previous section, Previously_Insured and Vehicle_Damage indicators are indeed the best predictor for customers' interests.
Random Forest Classifier:
![image](https://user-images.githubusercontent.com/77747784/122758969-13ebca00-d2b7-11eb-8dc2-b5fd4cc994de.png)
![image](https://user-images.githubusercontent.com/77747784/122759096-2f56d500-d2b7-11eb-830f-4958574b4ed8.png)
![image](https://user-images.githubusercontent.com/77747784/122759149-3da4f100-d2b7-11eb-9e86-e5767d9bebc5.png)
![image](https://user-images.githubusercontent.com/77747784/122759213-52818480-d2b7-11eb-9727-ce2a11ebd879.png)
Conclusion:Even though the precision in this model is higher than the logistic regression model, the recall and AUC-ROC score is considerably lower, meaning that the target market is not large enough, as shown by the confusion matrix. Only 9500 customers will receive ads compared to around 38000 by using logistic regression. Therefore, this model is not good enough to predict whether a customer is interested.



