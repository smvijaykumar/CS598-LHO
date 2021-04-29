
<br />
<p align="center">
  <h3 align="center">Drug Reviews Sentiment Analysis using Deep Learning Networks with Attention Models</h3>

  <p align="center">
    Description
    <br />
    <a href="https://github.com/smvijaykumar/CS598-LHO"><strong>Explore the docs »</strong></a>
    <br />
    <br />
 
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Sentiment analysis is a popular Natural Language Processing (NLP) technique to mine user perception and preference from real-time social media text. This same technique can be used in drug review texts to predict positive and negative sentiment. Hence, it helps to discover drug effectiveness, side effects and interaction of multiple drugs. It is especially helpful to detect sentiment from newly released drugs, e.g. COVID-19 vaccines. Most of the current sentiment analysis techniques are statistical learning (e.g. SVN, logistic regression, random forest) and basic deep neural network architecture (e.g. ANN, CNN and RNN). Their result is focus on test result accuracy rather than interpretability. Some also train one model per single medical condition to increase test accuracy. Our project uses a drug review text dataset with 215,063 reviews from UCI Machine Learning repository. Our project focuses on deep learning networks. We propose generic sentiment classifiers for all medical conditions in training dataset to use 1-D CNN, Bi-GRU and Transformer with attention to predict sentiment from drug review texts.. We use pre-trained word embedding GloVe to encode related words and concepts closer together. The results so far at this stage is encouraging. We have implemented base models, and base plus additional features at this point. All of the implemented models reach over 80% test accuracy. Additional input features extracted from NLP do improve deep learning networks prediction power. We will implement attention mechanisms to see if interpretability can be achieved without sacrificing accuracy.


### Built With

* [Python 3.7]()
* [Pytorch]()
* [Scikit-Learn]()
* [NLTK]()
* [Spacy]()
* [Pandas]()
* [Matplotlib]()
* [Wordcloud]()
* [W3lib]()
* [Textblob]()
* [tqdm]()
* [IPython]()



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This project was built with Anaconda distribution. For ease of use and setup the local environment quickly, use below environment export file.

To import anaconda environments, you can do it from anaconda navigator

**Step 1:**  Open anaconda navigator and click on environments

**Step 2**: Click on import located on the bottom of the screen.

![](https://i0.wp.com/evidencen.com/wp-content/uploads/2020/07/image-4.png?resize=302%2C85&ssl=1)

**Step 3**: Give the environment a new name,

-   Click the folder icon and select the  **environment.yml** file you exported in the last section
-   Then click import
-   Wait a few minutes and the environment will be imported along with it’s dependencies.

![](https://i2.wp.com/evidencen.com/wp-content/uploads/2020/07/image-5.png?resize=457%2C191&ssl=1)


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/smvijaykumar/CS598-LHO.git
   ```
2. The project files are under **sentiment-analysis-drug-reviews** folder



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_


<!-- CONTACT -->
## Contact

Vijayakumar Sitha Mohan - vs24@illinois.edu
Waitong Matthew Leung - wleung2@illinois.edu

Project Link: [https://github.com/smvijaykumar/CS598-LHO](https://github.com/smvijaykumar/CS598-LHO)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Multiclass-text-classification][https://www.kaggle.com/mlwhiz/multiclass-text-classification-pytorch](https://www.kaggle.com/mlwhiz/multiclass-text-classification-pytorch)
* [Sentiment Analysis in Pytorch](https://github.com/bentrevett/pytorch-sentiment-analysis)
* [IMDB Sentiment Analysis using Pytorch](https://github.com/iArunava/IMDB-Sentiment-Analysis-using-PyTorch)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/smvijaykumar/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/smvijaykumar/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/smvijaykumar/repo.svg?style=for-the-badge
[forks-url]: https://github.com/smvijaykumar/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/smvijaykumar/repo.svg?style=for-the-badge
[stars-url]: https://github.com/smvijaykumar/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/smvijaykumar/repo.svg?style=for-the-badge
[issues-url]: https://github.com/smvijaykumar/repo/issues
[license-shield]: https://img.shields.io/github/license/smvijaykumar/repo.svg?style=for-the-badge
[license-url]: https://github.com/smvijaykumar/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/smvijaykumar



