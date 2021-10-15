$$
\begin{align*}
\hat{\beta}_{\rm MAP}&=\mathop{\arg\max}_{\beta}\ P(\boldsymbol{y|X};\boldsymbol{\beta})P(\boldsymbol{\beta})\\
&=\mathop{\arg\max}_{\beta}\ \log P(\boldsymbol{y|X};\boldsymbol{\beta})+\log P(\boldsymbol{\beta}) \\
&=\mathop{\arg\max}_{\beta}\ \sum\limits_{i=1}^m\left[\log P(y_i|x_i;\beta_i)+\log P(\beta_i)\right]\\
&=\mathop{\arg\max}_{\beta}\ \sum\limits_{i=1}^m[-\frac{(y_i-\boldsymbol{\beta}^\mathrm{T}x_i)^2}{2\sigma^2}-\tau |\beta_i|]\\
&=\mathop{\arg\min}_{\beta}\ \sum\limits_{i=1}^m[(y_i-\boldsymbol{\beta}^\mathrm{T}x_i)^2+2\sigma^2\tau |\beta_i|]\\
&=\mathop{\arg\min}_{\beta}\ \sum\limits_{i=1}^m(y_i-\boldsymbol{\beta}^\mathrm{T}x_i)^2+\lambda ||\boldsymbol{\beta||}_1 \quad (\lambda=2\sigma^2\tau)
\end{align*}
$$

