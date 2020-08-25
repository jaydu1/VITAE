## Plot figures like [Dynplot](https://github.com/dynverse/dynplot) for visualization

- First use the function of scTGMVAE to create the needed elements for plot after training
```python
model.plot_output(init_node, gene=None)
```
- Move the generated files to designated `path`
- Then use the function `prepare_traj()` to import the elements (if `gene_name` is `NULL`, then `gene_expression` is not imported)
```R
trajectory = prepare_traj(path, root_milestone_id, gene_name)
```
- Finally use the functions in `dynplot` to plot the figures, e.g.
```R
pdf('dimred.pdf', width = 16, height = 8, onefile = F)
patchwork::wrap_plots(
  plot_dimred(trajectory, dimred = trajectory$dimred) + ggtitle('Cell ordeing'),
  plot_dimred(trajectory, color_cells = 'grouping', grouping = trajectory$grouping, dimred = trajectory$dimred) + ggtitle('True grouping'),
  plot_dimred(trajectory, color_cells = 'feature', feature_oi = 'G15', dimred = trajectory$dimred) + ggtitle('Gene expression'),
  plot_dimred(trajectory, color_cells = 'pseudotime', pseudotime = trajectory$pseudotime, dimred = trajectory$dimred) + ggtitle('Pseudotime')
)
dev.off()
```
Remember to specify the elements used for plotting, e.g. pseudotime
