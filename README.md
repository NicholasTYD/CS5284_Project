# CS5284_Project

Project Motivation

Advertising campaigns involve significant costs, and not every ad appeals to every audience. Poorly
targeted campaigns can inflate Cost-Per-Lead (CPL) and waste spending. By accurately identifying
and targeting the right audience segments, advertisers can optimize their ad spend and achieve
greater returns on investment.

Description

We plan to use historical campaign-level data (e.g., target audience attributes, impressions, reach,
channel, geography, creative, spend) joined with outcomes such as leads and realized CPL. The task
is to first learn feature segments in which CPL dispersion is small. Secondly, we then train a classifier
that can flag low-CPL leads or campaign settings in advance. The end product is a small set of
segments ranked by expected CPL and a scoring function that prioritizes low-CPL opportunities for
future campaigns. In short, we develop a predictive model that reduces marketing spending by finding
clusters with consistently low CPL and by predicting which incoming leads will be low-CPL.

Proposed Solution

We obtain a dataset containing information about advertisements (E.g. Target Audience, Impressions,
Reach, Expenditure). We train a neural network containing both linear layers to extract an embedding
of the dataset and a graph clustering algorithm to group the embedded data according to the distance
metric.

We run a baseline model using classical clustering algorithms (Kmeans/SVM etc).First, we’ll build
quick, interpretable baselines: treat decision-tree leaves as clusters (so each leaf is a small group with
low CPL spread), run k-means where we append a scaled CPL term to the features to nudge clusters
toward CPL-homogeneity, and train SVM to classify “low-CPL vs not”. In parallel, we’ll sketch the data
as a k-NN graph and use degree distributions, connected components, mutual-kNN pruning, and
simple 2-D embeddings to pick sensible normalizations and neighborhood sizes.

Once the baselines are in place, we’ll train the neural part end-to-end. A small MLP encoder turns the
tabular inputs X into embeddings Z. From Z we build a sparse k-NN affinity graph (RBF or cosine) and
add a soft clustering head that outputs assignment probabilities P. We optimize the encoder and head
jointly with; a relaxed normalized-cut objective to encourage clean partitions on the graph; a
CPL-aware penalty that shrinks within-cluster CPL dispersion (computed with the soft assignments
P); and a lightweight classification head on Z for low-CPL prediction. We’ll calibrate that classifier and
choose the operating threshold to minimize expected CPL, not just maximize accuracy. To keep
training stable and practical, we can try avoiding hard, non-differentiable steps (no argmax or hard
k-means in the loop), use temperature-controlled soft assignments, and if eigenvectors get flaky stick
to the relaxed formulation with a D-orthogonality penalty instead of backpropagating through an
eigendecomposition. We’ll also enforce a minimum cluster size, add tiny Laplacian jitter, and keep the
temperature > 0 so the model doesn’t collapse.
