# Differential Gaussian Rasterization

Used as the rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". If you can make use of it in your own research, please be so kind to cite us.

## Install:
```
git clone https://github.com/coder-bryan-123/diff-gaussian-rasterization.git
git submodule init
git submodule sync --recursive
cd diff-gaussian-rasterization
pip insatll -e .

# or
git clone --recursive https://github.com/coder-bryan-123/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
pip insatll -e .
```
## Usage:
```
python tests/forward.py
```

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>
