library(ggplot2)

# Multiple plot function
#
# ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
# - cols:   Number of columns in layout
# - layout: A matrix specifying the layout. If present, 'cols' is ignored.
#
# If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
# then plot 1 will go in the upper left, 2 will go in the upper right, and
# 3 will go all the way across the bottom.
#
multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  require(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


# eta = 0.001
eta001 <- read.table("avg_loss_0.001_0.0.txt", col.names=c("avg_loss"))
eta001$x <- c(1:23358)

p001 <- ggplot(eta001, aes(x, avg_loss))
p001 <- p001 + geom_line()
p001 <- p001 + theme_bw(base_size = 12, base_family = "")

p001 <- p001 + xlab("100 steps")
p001 <- p001 + ylab("Average Loss")
p001 <- p001 + ggtitle("Average Loss vs steps for eta = 0.001")


# eta = 0.01
eta01 <- read.table("avg_loss_0.01_0.0.txt", col.names=c("avg_loss"))
eta01$x <- c(1:23358)

p01 <- ggplot(eta01, aes(x, avg_loss))
p01 <- p01 + geom_line()
p01 <- p01 + theme_bw(base_size = 12, base_family = "")

p01 <- p01 + xlab("100 steps")
p01 <- p01 + ylab("Average Loss")
p01 <- p01 + ggtitle("Average Loss vs steps for eta = 0.01")


# eta = 0.05
eta05 <- read.table("avg_loss_0.05_0.0.txt", col.names=c("avg_loss"))
eta05$x <- c(1:23358)

p05 <- ggplot(eta05, aes(x, avg_loss))
p05 <- p05 + geom_line()
p05 <- p05 + theme_bw(base_size = 12, base_family = "")

p05 <- p05 + xlab("100 steps")
p05 <- p05 + ylab("Average Loss")
p05 <- p05 + ggtitle("Average Loss vs steps for eta = 0.05")


multiplot(p001, p01, p05, cols=1)


