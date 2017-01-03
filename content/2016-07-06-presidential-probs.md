Title: Fun With Presidential Probabilities
Slug: presidential-probabilities
Date: 2016-07-06 23:00
Category: Blog

Please don't try to read into this blog post any political beliefs or agenda. This is simply an exercise in conditional probabilities.

In case you're on the internet for the first time this year, this election has been... different. For example, there's talk about how GOP convention delegates might find a way to nominate somebody not named Donald Trump in Cleveland. And until recently a small minority thought there was a chance that Hillary Clinton's legal issues would preclude her from running. Which raises the question - **how would a different Democratic or Republican candidate do if they suddenly became the nominee?**

We can try to answer this questions using the implied probabilities from [Betfair's US election markets](https://www.betfair.com/exchange/plus/#/politics/market/1.107373419. I'm going to make one important assumption - that these betting odds accurately reflect probabilities. I'm not making this assumption because it's necessarily true, I'm making it because it makes this analysis more fun.

Betfair has markets for the party that will win, who will win each party's nomination, and which person will win the general election. One interesting note on that last market - the rules are very carefully written, and state that:
>This market will be settled according to the candidate that has the most projected Electoral College votes won at the 2016 presidential election. Any subsequent events such as a ‘faithless elector’ will have no effect on the settlement of this market. In the event that no Presidential candidate receives a majority of the projected Electoral College votes, this market will be settled on the person chosen as President in accordance with the procedures set out by the Twelfth Amendment to the United States Constitution.

Betfair has good lawyers.

## The Problem, Restated
We want to find out the probability of a Republican win given anybody but Trump, or a Democratic win given anybody but Hillary.

Remember that the probability of a Republican win can be written as 
```
P(R) = P(R | Trump)*P(Trump) + P(R | ~Trump)*P(~Trump)
P(R) = P(R & Trump) + P(R | ~Trump)*P(~Trump)
```
Where `P(R)` denotes the probability of a Republican win, `P(Trump)` is the probability of a Trump nomination for the GOP ticket (but not necessarily a Trump win), `P(R | Trump)` is the chance that the Republicans win *conditioned* on the event that Trump is their nominee, `P(R & Trump)` is the chance that both Trump is nominated and the GOP wins the Presidency, and `~Trump` denotes the event where Trump is not the nominee. Similarly, we have:
```
P(D) = P(D | Clinton)*P(Clinton) + P(D | ~Clinton)*P(~Clinton) 
P(D) = P(D & Clinton) + P(D | ~Clinton)*P(~Clinton)
```

After solving for the quantities we actually care about, we get 
```
P(R | ~Trump) = (P(R) - P(R & Trump)) / P(~Trump) 
```
```
P(D | ~Clinton) = (P(D) - P(D & Clinton)) / P(~Clinton)
```
Now we just need to fill in those numbers.

Betfair gives us decimal odds, and to find the implied probabilities all we have to do is take the inverse of the odds. The last trades (as of around 11pm EDT on July 6th) made for Hillary and Trump to be elected President were 1.37 and 4.3 respectively, which translates into 73.0% and 23.3% probabilities for `P(D & Clinton)` and `P(R & Trump)`. I'm making the assumption that they can only become President through their respective parties, which is technically false but extremely close to the truth.

For `P(D)` and `P(R)` we get 74.6% and 25.3% from the last trade on the winning party market. This market unfortunately has low trading volume and a large spread on the Republican side (23.3% - 25.6%), so its estimates are less trustworthy. 

And we can look at the intra-party nomination markets, where Hillary gets 97.1% and Trump gets 94.3%. That's 2.9% for `P(~Clinton)` and 5.7% for `P(~Trump)`.

To put it all together,
```
P(R | ~Trump) = (.253 - .233) / .057 = 35.1%
```
```
P(D | ~Clinton) = (.746 - .730) / .029 = 55.2%
```

Compare these with the conditional probabilities for the presumptive nominees: 
```
P(R | Trump) = P(R & Trump) / P(Trump) = .233 / .943 = 24.7%
```
```
P(D | Clinton) = P(D & Clinton) / P(Clinton) = .730 / .971 = 75.2%
```

## Conclusion (and why you shouldn't take this too seriously)

If the GOP delegates pull a #NeverTrump coup in Cleveland, they might expect to do better with their new candidate: a 35% conditional probability of a win versus 25%. Presumably the chaos and hurt feelings of such a coup are priced into this. This isn't the probability that some other candidate would have been better starting in May, it's the probability of a win given a switch in the near future.

If through some bizarre chain of events a non-Clinton took over this point, they'd be expected to do *worse* than she would in the general: 55% vs. 75%.

But to bring us back to reality, you should ignore the above analysis because:  
1. It's a debatable question whether you can actually turn betting odds into implied probabilities, given the house cut, bettors' risk avoidance preferences, etc.  
2. One of the markets I got data from has low volume and a large spread, and we therefore shouldn't take its predictions as seriously.  
3. Betfair only allows betting odds to be at discrete intervals, which means that rounding errors can compound.  
4. `P(~candidate)` refers to all possible other possible candidates in that party. Your favorite candidate might have a lower conditional probability to win than the hypothetical anonymous candidate.

This was mostly just a fun exercise in probability.