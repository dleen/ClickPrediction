lambda = c(0.0,0.002,0.004,0.006,0.008,0.010,0.012,0.014)

rmse = c(0.17334843748770765,
         0.17278782374238363,
         0.17277095177139887,
         0.1727846069254829,
         0.17280541432751717,
         0.17282746515285405,
         0.1728486797528293,
         0.17286844967719323)

baseline = c(0.173083735870231)
baseline <- rep(baseline, length(rmse))

p3b <- data.frame(lambda=c(lambda,lambda), rmse=c(rmse,baseline), 
                  RMSE=c(rep("rmse",length(rmse)),rep("baseline",length(rmse))))

p <- ggplot(p3b, aes(lambda, rmse, group=RMSE, linetype=RMSE))
p <- p + geom_line()
p <- p + theme_bw(base_size = 12, base_family = "")
p <- p + scale_linetype_manual(values=c("dashed", "solid"))

p <- p + xlab("lambda")
p <- p + ylab("RMSE")
p <- p + ggtitle("RMSE vs lambda for eta = 0.05")

p