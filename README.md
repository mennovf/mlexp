Some experiments in Neural Networks for learning. I use GenAI only for the peripheral features
such as display and inputs so I can actually learn something.

## Borders
Small Neural Network to learn a border. I.e. it classifies a point $P = (x, y)$ as being either inside or outside.

$$
(\textrm{softmax} \circ f)\left(
\begin{bmatrix}
x \\
y
\end{bmatrix}
\right) =
\begin{bmatrix}
p(\textrm{inside}) \\
p(\textrm{outside}))
\end{bmatrix}
$$

The `text.html` file is a small (AI-generated) web app to get a list of characters for a given string and font.
A character consists of a list of points which approximate a closed loop with linear segments.
These can be used to define an object function "border" with the polygons.InsidePolygons object.

#### Demo
![smiley](https://github.com/user-attachments/assets/e803d0c6-af55-4605-ab07-1482cdf24cb3)
