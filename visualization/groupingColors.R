check_groups <- function(grouping, groups) {
  if (is.null(groups) || !("color" %in% names(groups))) {
    groups <- tibble(
      group_id = unique(grouping),
      color = milestone_palette("auto", length(group_id))
    )
  }
  groups
}

sort_by_hue <- function(hex_colors) {
  hues <- grDevices::rgb2hsv(grDevices::col2rgb(hex_colors))[1, ]
  hex_colors[order(hues)]
}




#' @importFrom RColorBrewer brewer.pal
#' @importFrom grDevices rainbow rgb2hsv col2rgb
#' @importFrom rje cubeHelix
milestone_palette_list <- list(
  cubeHelix = function(n) rje::cubeHelix(n = n),
  Set3 = function(n) {
    cols <- RColorBrewer::brewer.pal(max(3, n), "Set3")[seq_len(n)]
    sort_by_hue(cols)
  },
  rainbow = function(n) grDevices::rainbow(n = n),
  auto = function(n) {
    if (n <= 12) {
      milestone_palette_list$Set3(n)
    } else {
      # milestone_palette_list$cubeHelix(n)
      all_colors <- grDevices::colors()[grep('gr(a|e)y', grDevices::colors(), invert = T)]
      all_colors <- sort_by_hue(all_colors)[-c(1:2)] # sort and remove white/black
      ix <- ceiling(seq(0, length(all_colors), length(all_colors)/(n+1)))
      all_colors[head(ix, -1)]
    }
  }
)

#' @param name The name of the palette. Must be one of \code{"cubeHelix"}, \code{"Set3"}, or \code{"rainbow"}.
#' @param n The number of colours to be in the palette.
#'
#' @rdname get_milestone_palette_names
milestone_palette <- function(name, n) {
  milestone_palette_list[[name]](n)
}

#' Get the names of valid color palettes
#'
#' @keywords plot_helpers
#'
#' @export
get_milestone_palette_names <- function() {
  names(milestone_palette_list)
}

calculate_hex_coords <- function(cell_positions, hex_cells) {
  xrange <- range(cell_positions$comp_1)
  yrange <- range(cell_positions$comp_2)
  
  # expand the smallest range so that both are equal
  shape <- diff(xrange) / diff(yrange) * sqrt(3) / 2 * 1.15
  if(shape > 1) {
    yrange <- c(yrange[1], yrange[2] + diff(yrange) * (shape - 1))
  } else {
    xrange <- c(xrange[1], xrange[2] + diff(xrange) * (1/shape - 1))
  }
  
  hexbin <- hexbin::hexbin(
    cell_positions$comp_1,
    cell_positions$comp_2,
    IDs = T,
    xbins = hex_cells,
    xbnds = xrange,
    ybnds = yrange,
    shape = 1
  )
  xy <- hexbin::hcell2xy(hexbin, check.erosion = FALSE)
  
  cell_positions$bin <- hexbin@cID
  bin_positions <- cell_positions %>%
    group_by(bin) %>%
    summarise(color = last(color)) %>%
    mutate(
      comp_1 = xy$x[match(bin, hexbin@cell)],
      comp_2 = xy$y[match(bin, hexbin@cell)]
    )
  
  hexcoords <- hexbin::hexcoords(
    diff(hexbin@xbnds)/hexbin@xbins / 2,
    diff(hexbin@xbnds)/hexbin@xbins / sqrt(3) / 2
  )
  
  hex_coords <- tibble(
    comp_1 = rep.int(hexcoords$x, nrow(bin_positions)) + rep(bin_positions$comp_1, each = 6),
    comp_2 = rep.int(hexcoords$y, nrow(bin_positions)) + rep(bin_positions$comp_2, each = 6),
    group = rep(seq_len(nrow(bin_positions)), each = 6),
    color = bin_positions$color[group]
  )
  
  hex_coords
}