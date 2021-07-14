## Hand Writing Number Classification
### About the project    
Use MNIST dataset which content 60,000 training and 10,000 testing images to train deep learning models, 
and adjust the model's architecture and hyperparameters to obtain the highest accuracy.  

### Data Processing


### Data Modeling

- CNN  
- MLP  

### Data Training & Testing  
Apply 20% training data as validation data and train 10 times to get the final accuracy of each model.  
After the optimized models, the accuracy of each model is as following:  
- CNN: 99.14%  
![CNN][product-screenshot0] 

- MLP: 97.68%  
![MLP][product-screenshot4] 

Check loss plots ensure there's no overfitting problem.  
- CNN  
![CNN][product-screenshot1]  

- MLP    
![MLP][product-screenshot5] 

### Evaluation  
Plot Confusion Matrix to see prediction result & Analyze the misclassified numbers.  
- CNN  
![MLP][product-screenshot2]  

- MLP  
![MLP][product-screenshot6]  


### Build With
* [scikit-learn](https://scikit-learn.org/stable/#)
* [matplotlib](https://matplotlib.org/)
* [keras](https://keras.org/)


<!-- CONTACT -->
## Contact

Yu-Chieh Wang - [LinkedIn](https://www.linkedin.com/in/yu-chieh-wang/)  
email: angelxd84130@gmail.com


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

[license-shield]: https://img.shields.io/github/license/angelxd84130/DeepLearning.svg?style=for-the-badge
[license-url]: https://github.com/angelxd84130/DeepLearning/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/yu-chieh-wang/
[product-screenshot0]: data/CNN_Accuracy.png
[product-screenshot1]: data/CNN_loss.png
[product-screenshot2]: data/CNN_ConfusionMatrix.png
[product-screenshot3]: data/CNN_plot.png
[product-screenshot4]: data/MLP_Accuracy.png
[product-screenshot5]: data/MLP_loss.png
[product-screenshot6]: data/MLP_ConfusionMatrix.png
[product-screenshot7]: data/MLP_plot.png
