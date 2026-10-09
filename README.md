# Variational Autoencoder in NumPy
After doing quite a bit of SFT on small LLMs for a research project, I realized that I didn't have much idea of how training worked under the hood. The training behind modern LLMs that I take for granted are built on top of years of r&d on learning paradigms, matrix factorization, synthetic data, autograd, ADAM, etc. 
A VAE was the first thing I built from a tutorial when starting to learn PyTorch with only my laptop cpu. The goal is this project is to develop a variational autoencoder (VAE) from scratch in NumPy with <span style="color:#eeeeff">**no AI**</span> and <span style="color:#red">**no reference code**</span> . All matrices and operations are handled with NumPy arrays and native matrix multiplication. 

##Complete list of all resources used in this project
#   Feed-forward networks: Michael Nielsen, Neural Networks and Deep Learning, Ch. 1 and Ch. 2, without reading reference code examples
#   Gradients and backpropagation: 
#       -> 3Blue1Brown: Backpropagation, intuitively / Backpropagation calculus
#       -> ritvikmath: "Backpropogation: Data Science Concepts"
#       -> https://tutorial.math.lamar.edu/classes/calci/linearapproximations.aspx
#       -> Softmax function: https://en.wikipedia.org/wiki/Softmax_function / https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
#   SGD:
#       -> https://en.wikipedia.org/wiki/Stochastic_gradient_descent 
#       -> https://www.geeksforgeeks.org/deep-learning/adam-optimizer/
#       -> Jacob Murri (UCLA), Math 118 Lecture Notes 
#       -> Brendan Martin (UCLA), discussion on ADAM 
#
#   Autoencoders:
#       -> Durk Kingma, Max Welling: An Introduction to Variational Autoencoders 
#               (I had to do a lot of reading to understand what they were saying, since I've never taken Bayesian stats)
#       -> https://apxml.com/courses/introduction-autoencoders-feature-learning/chapter-3-how-autoencoders-learn/autoencoder-loss-functions
#       -> https://medium.com/retina-ai-health-inc/variational-inference-derivation-of-the-variational-autoencoder-vae-loss-function-a-true-story-3543a3dc67ee
#       -> https://stats.stackexchange.com/questions/323568/help-understanding-reconstruction-loss-in-variational-autoencoder

