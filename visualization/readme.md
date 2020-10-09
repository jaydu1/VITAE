## Plot figures like [Dynplot](https://github.com/dynverse/dynplot) for visualization

- First use the function of scTGMVAE to create the needed elements for plot after training
```python
model.plot_output(init_node, gene=None, batchsize = 32, cutoff=None, gene=None, thres=0.5, method='mean')
```
- Move the generated files to designated `path`
- Then use the function `prepare_traj()` to import the elements (if `gene_name` is `NULL`, then `gene_expression` is not imported)
```R
trajectory = prepare_traj(path, root_milestone_id, gene_name)
```
- Finally use the functions in `dynplot` to plot the figures, e.g. if `gene_name = G15` is given
```R
pdf('dimred.pdf', width = 16, height = 8, onefile = F)
patchwork::wrap_plots(
  # Cell ordering
  plot_dimred(trajectory, dimred = trajectory$dimred) + ggtitle('Cell ordeing'),
  # Grouping
  plot_dimred(trajectory, color_cells = 'grouping', grouping = trajectory$grouping, dimred = trajectory$dimred) + ggtitle('True grouping'),
  # Gene expression
  plot_dimred(trajectory, color_cells = 'feature', feature_oi = 'G15', dimred = trajectory$dimred) + ggtitle('Gene expression'),
  # Pseudotime
  plot_dimred(trajectory, color_cells = 'pseudotime', pseudotime = trajectory$pseudotime, dimred = trajectory$dimred) + ggtitle('Pseudotime')
  # w_tilde posterior variance (use the approach for 'pseudotime')
  plot_dimred(trajectory, color_cells = 'pseudotime', pseudotime = trajectory$pos_var, dimred = trajectory$dimred, hex_cells = FALSE) +
  viridis::scale_color_viridis("posterior_variance", option = 'plasma') + ggtitle('Posterior variance')
)
dev.off()
```
Remember to specify the elements used for plotting, e.g. pseudotime
