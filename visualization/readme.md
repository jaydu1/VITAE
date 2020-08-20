## Plot figures like [Dynplot](https://github.com/dynverse/dynplot) for visualization

- First use the function of scTGMVAE to create the needed elements for plot after training
```python
model.plot_output(init_node, gene=None)
```
- Move the generated files to designated `path`
- Then use the function `IO(path, gene_name=NULL)` to import the elements (if `gene_name` is `NULL`, then `gene_expression` is not imported)
```R
trajectory = IO(path, gene_name)
```
- Finally use the function `plott(trajectory, plotwhat)` to plot the figures, e.g.
```R
pdf('combine.pdf', width = 16, height = 8, onefile = F)
patchwork::wrap_plots(
  plott(trajectory, 'milestone') + ggtitle('Cell ordeing'),
  plott(trajectory, 'grouping') + ggtitle('True grouping'),
  plott(trajectory, 'expression') + ggtitle('Gene expression'),
  plott(trajectory, 'pseudotime') + ggtitle('Pseudotime')
)
dev.off()
```
