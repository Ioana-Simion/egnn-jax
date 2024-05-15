# **DEMETAr: Double Encoder Method for an Equivariant Transformer Architecture"**

### _I. Simion, S. Vasilev, J. Schäfer, G. Go, T. P. Kersten_

---

In this blog post, we discuss and extend on the findings of the PMLR 2021 paper titled ["E(n) Equivariant Graph Neural Networks"](http://proceedings.mlr.press/v139/satorras21a/satorras21a.pdf). This paper introduces a more efficient way of implementing equivariance in graph neural networks, which we adapt for transformer architectures. While equivariant transformers do already exist, we propose a novel method of leveraging these equivariances which plays into the transformer's strengths.

This blogpost serves three purposes: 
1. Explain the fundamental ideas of equivariance for neural networks introduced by Satorras et al. (2021).
2. Verify the authors' claims by reproducing their results.
3. Give an overview of our transformed-based method.

---

## **Equivariance in Neural Networks**

As equivariance is prevalent in the natural sciences \[1, 2, 3\], it makes sense to utilize them for our neural networks. This is not a straightforward process, however, as encoding the positions directly does not guarantee that the network learns these geometric patterns properly and efficiently [citation here]. Because of this, most methods use either invariant geometric information (i.e., between-point distances [citation here]) or covariant information (i.e., with steerable functions [citation here]). 

With an origin spanning a few decades, Graph Neural Networks (GNNs) were introduced which bridged the gap between deep learning and graph processing \[4\]. However, compared to other similar methods at the time, Satorras et al. (2021) introduce the idea of inputting the relative squared distance between two coordinates into the edge operation. This allows their method to bypass any expensive computations/approximations in comparison while retaining high performance levels.

To test the performance of their algorithm, we reproduce their main results, both qualitative and quantitative, on the 
QM9 [citation here] dataset.


## **<a name="equiv">Equivariance</a>**

Given a set of $T_g$ transformations on $X$ ($T_g: X \rightarrow X$) for an abstract group $g \in G$, a function $\varphi: X \rightarrow Y$ is equivariant to $g$ if an equivalent transformation exists on its output space $S_g: Y \rightarrow Y$ such that:

$$\begin{align} 
\varphi(T_g(x)) = S_g(\varphi(x)). \qquad \qquad \text{(Equation 1)}
\end{align}$$

In other words, translating the input set $T_g(x)$ and then applying $\varphi(T_x(x))$ on it yields the same result as first running the function $y = \varphi(x)$ and then applying an equivalent translation to the output $T_g(y)$ such that Equation 1 is fulfilled and $\varphi(x+g) = \varphi(x) + g$ \[5\].

To guarantee this property, ...

<!-- <table align="center">
  <tr align="center">
      <td><img src="figures/aprox.png" width=800></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 1.</b> The Markov process of diffusing noise and denoising [5].</td>
  </tr>
</table> -->


## **<a name="discover">Equivariant Graph Neural Networks</a>**

The standard GNN implementation is generally the following:


In order to make this implementation equivariant, we need to  

## **<a name="architecture">Equivariant Transformers</a>**

Our method of improving this architecture would be to leverage the capabilites of transformers \[6\]. The key difference between these and GNNs is that the former treats the entire input as a fully-connected graph. This would typically make transformers less-suited , though many papers have been published which demonstrate their effectivity in handling these tasks \[7\]. 

As our contribution to the field, we introduce a dual encoder system. The first one contains all the node features and normalized distances to the molecule's center of mass, while the other exclusively encodes the edge features (i.e., bond type) and an edge length feature. 

To explain this approach, we first need to define the following components:

$$\begin{align} 
& K^l_e, V^l_e &: \text{the keys, values of edge features at layer } l. \\
& K^l_n, V^l_n, Q^l_n &: \text{the keys, values of node features at layer } l.
\end{align}$$

Now we can begin with the actual approach. We first use an edge encoder with $p$ transformer layers on the data to transform the edge features into the node space. Then, we obtain $K^p_e$, $V^p_e$ and perform the following attention operation:

$$\begin{align} 
Z^p_e = softmax(Q^p_e K^{pT}_n + M) V^p_n / d, \qquad \qquad \text{(Equation 2)}
\end{align}$$

<!-- ^Why is it only d here? - Greg -->

where the output $Z^p_e$ is a matrix of size $n \times d$ (due to the cross-attention) which contains edge encoded information in the node space for every node and $M$ is an adjacency matrix mask of size $n \times e$ where all connections are 0's and non-connections are $-\infty$ to prohibit the attention from attending to non-connected edges. Furthermore, for all layers $< p$, only the edge queries, keys, and values are used, thus no mask is required here. Meanwhile, in the $p$-th layer, we limit the attention to only the connected nodes to calculate the edge features for every node in order to use the node keys. Lastly, the final division after softmaxing by $d$ is to normalize the output scale, a method employed by most other transfomer architectures.

Now, we need to obtain the node encodings, which is done through the following: 

$$\begin{align} 
Z^r_n = softmax(Q^r_n K^{rT}_n) V^n_r / d, \qquad \qquad \text{(Equation 3)}
\end{align}$$

where $Z^r_n$ is the output of layer $r$, which is the encoder's last layer.

Now that we have both the node and edge features encoded, we can simply sum these encodings to combine them together:

$$\begin{align} 
Z^0_j = Z^p_e + Z^r_n, \qquad \qquad \text{(Equation 4)},
\end{align}$$

where $Z^0_j$ is the input for a join encoder $Z^j$. This operation can alternatively be interpreted as a residual connection in the node space, where $Z^r_n$ is the residual connection. After this operation, we continue the computation with an $h$-layer joint encoder and get the output $Z^h_j$. One final note is that we have a [CLS] token which is used for classification in the $Z^0_j$ or the $Z^0_n$ input.


## **<a name="architecture">Evaluating the Models</a>**

To evaluate the performance of the models, ...

## **<a name="reproduction">Reproduction of the Experiments</a>**

...

<table align="center">
  <tr align="center">
      <th align="left">Label</th>
      <th align="left">$y_{ref}$</th>
      <th align="left">$y_{target}$</th>
      <th>$\lambda_{\text{CLIP}}$</th>
      <th>$t_{\text{edit}}$</th>
      <th>Domain</th>
  </tr>
  <tr align="center">
    <td align="left">smiling</td>
    <td align="left">"face"</td>
    <td align="left">"smiling face"</td>
    <td>0.8</td>
    <td>513</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">sad</td>
    <td align="left">"face"</td>
    <td align="left">"sad face"</td>
    <td>0.8</td>
    <td>513</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">angry</td>
    <td align="left">"face"</td>
    <td align="left">"angry face"</td>
    <td>0.8</td>
    <td>512</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">tanned</td>
    <td align="left">"face"</td>
    <td align="left">"tanned face"</td>
    <td>0.8</td>
    <td>512</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">man</td>
    <td align="left">"a person"</td>
    <td align="left">"a man"</td>
    <td>0.8</td>
    <td>513</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">woman</td>
    <td align="left">"a person"</td>
    <td align="left">"a woman"</td>
    <td>0.8</td>
    <td>513</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">young</td>
    <td align="left">"person"</td>
    <td align="left">"young person"</td>
    <td>0.8</td>
    <td>515</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">curly hair</td>
    <td align="left">"person"</td>
    <td align="left">"person with curly hair"</td>
    <td>0.8</td>
    <td>499</td>
    <td>IN</td>
  </tr>
  <tr align="center">
    <td align="left">nicolas</td>
    <td align="left">"Person"</td>
    <td align="left">"Nicolas Cage"</td>
    <td>0.8</td>
    <td>461</td>
    <td>UN</td>
  </tr>
  <tr align="center">
    <td align="left">pixar</td>
    <td align="left">"Human"</td>
    <td align="left">"3D render in the style of Pixar"</td>
    <td>0.8</td>
    <td>446</td>
    <td>UN</td>
  </tr>
  <tr align="center">
    <td align="left">neanderthal</td>
    <td align="left">"Human"</td>
    <td align="left">"Neanderthal"</td>
    <td>1.2</td>
    <td>490</td>
    <td>UN</td>
  </tr>
  <tr align="center">
    <td align="left">modigliani</td>
    <td align="left">"photo"</td>
    <td align="left">"Painting in Modigliani style"</td>
    <td>0.8</td>
    <td>403</td>
    <td>UN</td>
  </tr>
  <tr align="center">
    <td align="left">frida</td>
    <td align="left">"photo"</td>
    <td align="left">"self-portrait by Frida Kahlo"</td>
    <td>0.8</td>
    <td>321</td>
    <td>UN</td>
  </tr>
  <tr align="left">
    <td colspan=7><b>Table 1.</b> Hyperparameter settings of reproducibility experiments. The "domain" column corresponds<br>to the attribute being in-domain (IN) or unseen-domain (UN).</td>
  </tr>
</table>

...


## **Concluding Remarks**

...

## **Authors' Contributions**

- Ioana: [fill in your contribution]
- Stefan: [fill in your contribution]
- Jonas: [fill in your contribution]
- Gregory: Code documentation, dependency setup, assisting with comparing implementations and searching for ideas, writing.
- Thies: [fill in your contribution]

## Bibliography

[1] Balaban, A. T. (1985). Applications of graph theory in chemistry. In Journal of Chemical Information and Computer Sciences, 25(3), pp. 334–343. American Chemical Society (ACS). https://doi.org/10.1021/ci00047a033 

[2] Gupta, P., Goel, A., Lin, J.J., Sharma, A., Wang, D., & Zadeh, R.B. (2013). WTF: the who to follow service at Twitter. Proceedings of the 22nd international conference on World Wide Web.

[3] Miller, G. A. (1995). WordNet. In Communications of the ACM (Vol. 38, Issue 11, pp. 39–41). Association for Computing Machinery (ACM). https://doi.org/10.1145/219717.219748 

[4] Gori, M., Monfardini, G., & Scarselli, F. (2005). A new model for learning in graph domains. Proceedings. 2005 IEEE International Joint Conference on Neural Networks, 2(2), 729-734.

[5] Satorras, V. G., Hoogeboom, E., & Welling, M. (2021). E(n) Equivariant Graph Neural Networks. In Proceedings of the 38th International Conference on Machine Learning, 139. https://doi.org/10.48550/ARXIV.2102.09844

[6] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., … Polosukhin, I. (2017). Attention is All you Need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), Advances in Neural Information Processing Systems, 30.

[7] Thölke, P., & De Fabritiis, G. (2022). Equivariant Transformers for Neural Network based Molecular Potentials. In International Conference on Learning Representations.