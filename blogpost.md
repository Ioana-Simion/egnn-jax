# **DEMETAr: Double Encoder Method for an Equivariant Transformer Architecture"**

### _I. Simion, S. Vasilev, J. Schäfer, G. Go, T. P. Kersten_

---

This blogpost serves as an introduction to our novel implementation of equivariance for transformer architectures. While equivariant transformers do already exist, we propose a method that utilizes two encoders for the node and edge information separately. This allows for more flexibility in the inputs we provide.

This blogpost serves three purposes: 
1. Explain the ideas of equivariance in transformer networks while also explaining some of the methods used.
2. Providing an overview of some reproduction results for other methods (i.e., the Equivariant Graph Neural Network).
3. Give an overview of our method and a comparison with the aforementioned reproduction results.

---

## **Equivariance in Neural Networks**

As equivariance is prevalent in the natural sciences \[1, 2, 3\], it makes sense to utilize them for our neural networks, especially given the evidence suggesting that it significantly improves performance through increasing the network's generalizability \[8\]. Because of this, various techniques have been created based on this idea \[9, 10, 11\] and are still being expanded upon. 

As a baseline, we compare our methods to varying architectures, with the first being the Graph Neural Network (GNN). It was introduced to bridge the gap between deep learning and graph processing as it operates within the graph domain \[4, 15, 16\]. Here, we compare our method to that of \[5\], as it is a GNN implementation which inputs the relative squared distance between two coordinates into the edge operation. This method bypasses any expensive computations/approximations in comparison while retaining high performance levels, making it preferable compared to most other GNN architectures.

For all the aforementioned methods, we evaluate and reproduce their performance on the QM9 \[12, 13\] and N-body \[14\] datasets. The former is used to evaluate the model performances on invariant tasks due to only requiring property predictions. Meanwhile, the latter is to test how well each model can handle equivariance in the data.


## **<a name="recap">Recap of Equivariance</a>**

Given a set of $T_g$ transformations on $X$ ($T_g: X \rightarrow X$) for an abstract group $g \in G$, a function $\varphi: X \rightarrow Y$ is equivariant to $g$ if an equivalent transformation exists on its output space $S_g: Y \rightarrow Y$ such that:

$$\begin{align} 
\varphi(T_g(x)) = S_g(\varphi(x)). \qquad \qquad \text{(Equation 1)}
\end{align}$$

In other words, translating the input set $T_g(x)$ and then applying $\varphi(T_x(x))$ on it yields the same result as first running the function $y = \varphi(x)$ and then applying an equivalent translation to the output $T_g(y)$ such that Equation 1 is fulfilled and $\varphi(x+g) = \varphi(x) + g$ \[5\].

<!-- <table align="center">
  <tr align="center">
      <td><img src="figures/aprox.png" width=800></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 1.</b> The Markov process of diffusing noise and denoising [5].</td>
  </tr>
</table> -->


## **<a name="gnns">Equivariant Graph Neural Networks</a>**

For a given graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with nodes $v_i \in \mathcal{V}$ and edges
$=e_{ij} \in \mathcal{E}$, we can define a graph convolutional layer as the following:

$$\begin{align} 
& \mathbf{m}\_{ij} &= \varphi_e (\mathbf{h}\_i^l, \mathbf{h}\_j^l, a_{ij}), \qquad \qquad \text{(Equation 2)} \\
& \mathbf{m}\_{i} &= \sum_{j \in \mathcal{N}\_i } \mathbf{m}\_j, \qquad \qquad \text{(Equation 3)} \\
& \mathbf{h}\_i^{l+1} &= \varphi_h (\mathbf{h}\_i^l, \mathbf{m}\_i), \qquad \qquad \text{(Equation 4)}
\end{align}$$

where $\mathbf{h}\_i^l \in \mathbb{R}^{nf}$ nf is the nf-dimensional embedding of node $v_i$ at layer l$$, $a_{ij}$ are the edge attributes, $\mathcal{N}\_i$ is the set of neighbors of node $v_i$, and $\varphi_e$ and $\varphi_h$ are the
edge and node operations respectively, typically approximated by Multilayer Perceptrons (MLPs).

In order to make this implementation equivariant, \[5\] introduced the inputting of the relative squared distances between two points and updating of the node positions at each time step, leading to the following formulae:

$$\begin{align} 
& \mathbf{m}\_{ij} &= \varphi_e (\mathbf{h}\_i^l, \mathbf{h}\_j^l, ||\mathbf{x}\_i^l - \mathbf{x}\_j^l||^2, a_{ij}), \qquad \qquad \text{(Equation 5)} \\
& x_i^{l+1} &= x_i^l + C \sum_{j \neq i} (\mathbf{x}\_i^l - \mathbf{x}\_j^l) (\mathbf{m}\_{ij}) \varphi_x  \qquad \qquad \text{(Equation 6)} \\
& \mathbf{m}\_{i} &= \sum_{j \in \mathcal{N}\_i } \mathbf{m}\_j, \qquad \qquad \text{(Equation 7)} \\
& \mathbf{h}\_i^{l+1} &= \varphi_h (\mathbf{h}\_i^l, \mathbf{m}\_i), \qquad \qquad \text{(Equation 8)}
\end{align}$$

This idea of using the distances during computation forms one of the bases of our proposed transformer architecture, as it is a simple yet effective way to impose geometric equivariance within a system.

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
Z^p_e = \frac{softmax(Q^p_e K^{pT}_n + M) V^p_n}{\sqrt(d)}, \qquad \qquad \text{(Equation 2)}
\end{align}$$

where the output $Z^p_e$ is a matrix of size $n \times d$ (due to the cross-attention) which contains edge encoded information in the node space for every node and $M$ is an adjacency matrix mask of size $n \times e$ where all connections are 0's and non-connections are $-\infty$ to prohibit the attention from attending to non-connected edges. Furthermore, for all layers $< p$, only the edge queries, keys, and values are used, thus no mask is required here. Meanwhile, in the $p$-th layer, we limit the attention to only the connected nodes to calculate the edge features for every node in order to use the node keys. Lastly, the final division after softmaxing by the dimension size $\sqrt(d)$ is to normalize the output scale, a method employed by most other transfomer architectures.

Now, we need to obtain the node encodings, which is done through the following: 

$$\begin{align} 
Z^r_n = \frac{softmax(Q^r_n K^{rT}_n) V^n_r}{\sqrt(d)}, \qquad \qquad \text{(Equation 3)}
\end{align}$$

where $Z^r_n$ is the output of layer $r$, which is the encoder's last layer. Also, similar to the previous formula, we also control the output magnitude by dividing by $\sqrt(d)$.

Now that we have both the node and edge features encoded, we can simply sum these encodings to combine them together:

$$\begin{align} 
Z^0_j &= Z^p_e + Z^r_n, \qquad \qquad \text{(Equation 4)}
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

[8] Bronstein, M.M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. ArXiv, abs/2104.13478.

[9] Cohen, T. & Welling, M. (2016). Group Equivariant Convolutional Networks. Proceedings of The 33rd International Conference on Machine Learning. https://proceedings.mlr.press/v48/cohenc16.html.

[10] Thomas, N., Smidt, T.E., Kearnes, S.M., Yang, L., Li, L., Kohlhoff, K., & Riley, P.F. (2018). Tensor Field Networks: Rotation- and Translation-Equivariant Neural Networks for 3D Point Clouds. ArXiv, abs/1802.08219.

[11] Maron, H., Litany, O., Chechik, G. & Fetaya, E. (2020). On Learning Sets of Symmetric Elements. roceedings of the 37th International Conference on Machine Learning. https://proceedings.mlr.press/v119/maron20a.html.

[12] Blum, L. C., & Reymond, J.-L. (2009). 970 million druglike small molecules for virtual screening in the chemical universe database GDB-13. Journal of the American Chemical Society, 131(25), 8732–8733. https://doi.org/10.1021/ja902302h 

[13] Montavon, G., Rupp, M., Gobre, V., Vazquez-Mayagoitia, A., Hansen, K., Tkatchenko, A., Müller, K.-R., & von Lilienfeld, O. A. (2013). Machine learning of molecular electronic properties in chemical compound space. New Journal of Physics, 15(9), 095003. http://stacks.iop.org/1367-2630/15/i=9/a=095003

[14] Kipf, T., Fetaya, E., Wang, K., Welling, M. & Zemel, R. (2018). Neural Relational Inference for Interacting Systems. Proceedings of the 35th International Conference on Machine Learning. https://proceedings.mlr.press/v80/kipf18a.html.

[15] Bruna, J., Zaremba, W., Szlam, A., & Lecun, Y. (2014). Spectral networks and locally connected networks on graphs. In International Conference on Learning Representations (ICLR2014), CBLS.

[16] Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. CoRR, abs/1609.02907. http://arxiv.org/abs/1609.02907