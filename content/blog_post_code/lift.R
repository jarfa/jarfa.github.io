library(ggplot2)
library(tidyr)
library(scales)

csv_file = "/Users/jarfa/Dropbox/jarfa.github.io/content/blog_post_code/lift.csv"
png_file = "/Users/jarfa/Dropbox/jarfa.github.io/content/images/lift/CI.png"

lift_plot = read.csv(csv_file, stringsAsFactors=FALSE) %>%
  gather(type, value, high, low) %>%
  ggplot(aes(x=CI, y=value, color=method)) + geom_point(alpha=0.5, size=0.6) +
  scale_x_reverse("Confidence", labels=scales::percent) +
  scale_y_continuous("C.I. for Lift(B | A)", labels=scales::percent) +
  geom_hline(aes(yintercept=0.5), lty=2) +
  geom_vline(aes(xintercept=0.95), lty=2, color='red') +
  theme_minimal()

ggsave(filename=png_file, lift_plot, width=7, height=5)
