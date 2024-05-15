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
\varphi(T_g(x)) = S_g(\varphi(x)) \qquad \qquad \text{(Equation 1)}
\end{align}$$

In other words, translating the input set $T_g(x)$ and then applying $\varphi(T_x(x))$ on it yields the same result as first running the function $y = \varphi(x)$ and then applying an equivalent translation to the output $T_g(y)$ such that Equation 1 is fulfilled and $\varphi(x+g) = \varphi(x) + g$ \[5\].

To guarantee this property, ...

<table align="center">
  <tr align="center">
      <td><img src="figures/aprox.png" width=800></td>
  </tr>
  <tr align="left">
    <td colspan=2><b>Figure 1.</b> The Markov process of diffusing noise and denoising [5].</td>
  </tr>
</table>


## **<a name="discover">Equivariant Graph Neural Networks</a>**

The standard GNN implementation is generally the following:


In order to make this implementation equivariant, we need to  

## **<a name="architecture">Equivariant Transformers</a>**

Our method of improving this architecture would be to leverage the capabilites of transformers [cite Vaswani again]. ...


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