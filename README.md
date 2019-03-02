Complementary GAN
===
An PyTorch implementation 

## result

### Mnist
sample from  0~20 epoch


### Cifar10
sample from  0~80 epoch

run code , 'sh run,sh' before modifying the configs of run.sh file

write the outlines in the following format:
 
# Introduction 

# Method
## Literature Review
* cGAN: Explaining conditional gan with the formulation
Generative adversarial networks learn a distribution over data $x$, which contains two adversarial players. Generator $G(z)$ is trained to map the $x$ from a prior random noise $z$, and discriminator  $D(x)$ predicts the probability of that $x$ comes from training data $x \sim p_{data}$ or generated data $G(z)\sim p_{z}$. The optimization form as follow:
$\min\limits_{G}\{\max\limits_{D}  \{V(D,G)\}\} = E_{x \sim p_{data}(x)}[logD(x)]+E_{z \sim p_{z}(z)}[1-logD(G(z))]$ 
To extend gans, we can simply plug conditional labels, like class label $y$, into generator $G$ and discriminator $D$. And the conditional generated data $G(z|y)$ and training data $x|y$ with known label $y$ can be fed into discriminator $D_(x|y)$ with an intermediate classifier $C$ which is implemented by ACGAN.   
$\min\limits_{G}\{\max\limits_{D}  \{V(D,G)\}\} +\max\limits_{C} \{C(y=c|x\sim p_{data}(x))\} + \max\limits_{G} \{C(y=c|G(z|y))\} $ 
* Complementary Learning: 
  - Complementary label $\overline{y}$ implies which label an image doesn't belong to. In practice, complementary labels are cheaper to notate than true label. The labeler only need to answer 'yes' or 'no' for an image with a given label, however it costs much more time to input the real label.  Assuming there are K classes in dataset $S= \\{ y_{i}|i=1 \cdots K \\}$, if complementary label is uniformly sampled from the rest $K-1$ false labels for each images, then $p(\overline{y})=p(y) \cdot 1/(K-1)$ 
   - The output of a classifier model given data $x$ is often a vector $\\{p(y=i|x),i=1 \cdots K\\}$ consisting of multiple binary probabilities for each class. Thus, there is a mapping between $p(\overline{y})$ and $p(y)$ by a transition matrix. 
$p(\overline{y}=j)=p(y=i) \cdot 1/(K-1)$, $i \not= j \Rightarrow p(\overline{y})=M^{T}p(y), M_{i,j}=p(\overline{y}=j|y=i)$
This transition matrix $M$ directly maps from $p(y)$ to $p(\overline{y})$, thus this method can plug the output the possibility $M^{T}p(y)$ and complementary label into a classification loss, such as cross entropy loss.

## Proposed method
* Complementary GAN
In our method, we hope to build a conditional synthesis model through complementary labels $\overline{y}$. Naively, we can simply apply complementary label $\overline{y}$ as condition and model $G(z|\overline{y})$, however it is impractical, we need map to $G(z|y) conditioned by true label$. Thus our proposed model need discriminator learns a true label classifier $C_{y}(y|x)$ rather than $C_{\overline{y}}(\overline{y}|x)$ in the training time which can help map generator to $G(z|y)$. Our model can be formulated as follows:
$\min\limits_{G}\{\max\limits_{D}  \{V(D,G)\}\} +\max\limits_{C_{\overline(y)}} \{C_{\overline{y}}(\overline{y}=\overline{c}|x\sim p_{data}(x))\} + \max\limits_{G} \{C(y=c|G(z|y))\} , C_{\overline{y}}=M^{T} \cdot C_{y}$
* Semi-Supervised Learning with unlabeled data
