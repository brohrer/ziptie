# Ziptie: An unsupervised feature creation algorithm

## What can it do?

![Level 5 Ziptie bundles from photos](/images/ziptie_level_5.png)

This is a few of the features created from 11x11 pixel patches pulled from color images.
Most of those pixels are gray, indicating that they aren't part
of the feature. Around 20-25 of them are black, white, or color,
showing the input pattern that maximizes the activation for that feature.
Many of them show paired light and dark pixels defining rough edges.
There are a variety of other shapes as well.
There are triples of light, dark, light pixels showing line segments;
curves; bright blobs; and other more intricate patterns.

The method here is inspired by Hebbian "what fires together wires together" learning.
It's an unsupervised clustering method for grouping channels (such as pixels in a camera)
together to make features.
I call it Ziptie, because the notion of bundling cables together gives a strong
intuitive head start to understanding how it works.

My favorite thing about Ziptie is that it is data agnostic. It created these edges and lines
and blobs and curves without knowing that these were pixels, or knowing where they were located
with respect to each other. The entire 3D image array was flattened before being passed in.

All of the structure in the features emerged organically, based on the patterns in the data.
What I like about this is that, unlike Convolutional Neural Networks, it can learn features
on any kind of data, even if it has no 2D or 3D structure.

I designed it to work with the heterogeneous data of robots: imagery, yes, but also
odometry, torque, range, audio, and myriad special purpose sensors.


## How it works

For the authoritative source, I've [liberally commented the code](https://gitlab.com/brohrer/ziptie/-/blob/main/ziptie/algo.py).
Here's a summary of the high points.

### Fuzzy categorical data

There is a non-standard funnel all the data has to pass through
before a Ziptie can start working with it.
It has to be converted to [fuzzy categorical data](https://e2eml.school/fuzzy_categoricals).
Every sensor needs to be converted to
a collection channels whose values vary between zero and one.

For pixels this is fairly natural. When scaled to [0, 1], a pixel's value *v*
is a fuzzy categorical variable representing the BRIGHT state.
Because of the quirks of fuzzy categoricals, we also have to create a variable
representing the DARK state of that pixel, *1 - v*.

### Bundling

Once all the data has been transformed into a big collection
of fuzzy categorical variables, it gets fed into a Ziptie.
There inputs that tend to be active at the same time get
clustered together. The math behind this is essentially counting.

The clustering in zipties is agglomerative. To use the metaphor of cables carrying signals,
each input is like a cable. The sequence of values it takes is its signal.
Sequential values can be related, as in time series data like audio or stock prices.
Or they can be independent, as in image classification benchmarks.

The ziptie algorithm finds cables that tend to be co-active, that is,
they tend to be positively valued at the same time.
Note that this is different than correlation.
Correlation is also strengthened when two signals are inactive at the same time.

The co-activation between two cables for a set of inputs is the product
of their two input values. Because all inputs come from fuzzy categorical variables,
they are known to be between zero and one, and so the product of any two inputs
will also be between zero and one. If a cable is inactive and has a value of zero,
then its co-activity with all others is zero.

To find patterns in activity, every channel's cables co-activity with every
other cable is calculated for each set of inputs, and they are summed over
time to find trends. Once the aggregated co-activity between a pair of
cables crosses the threshold, those two cables become bundled,
as if bound together by a plastic zip tie. This process continues,
and every time a cable pair exceeds the co-activity threshold for 
bundle creation, a new bundle is created.

The bundling process is conceptually similar to
[byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding),
where frequently co-occurring characters get represented
and replaced by a unique code
of their own. The process is repeated until even long strings that
occur often are represented by a single byte.


### Bundle activities

The output of a ziptie is the activity of each of its bundles.
A bundle's activity is calculated by taking the minimum value of the cables
that contribute to it. Because of this, bundle activities are also constrained
to be between zero and one. This makes them behave just like input cables.
And in fact, after they are created, a ziptie treats its bundles like additional
input cables, and find their co-activity with other cables.
In this way bundles can grow. Two cables can become bundled and then one by one
additional cables can be added creating a many-cable bundle of coactive cables.

The other quirk of calculating bundle activities is that after
the minimum value of a bundles cables is found and assigned
to the bundle activity, that value is subtracted from that 
of each of the cables. In this way, a cable's value is a finite
quantity that can only contribute once to a bundle's activity.
After it has contributed to a bundle, its activity is reduced by that amount.
Because of this, the order in which bundle activities are calculated
is important. The most recently created bundles' activities are calculated first.
These will be the bundles with the greatest number of cables.
They will be the most complex and if active, account for the most activity with a single value.

After all bundle activities have been calculated, a cable’s remaining
activity is available for contributing to co-activity calculations.
This prevents activity from getting counted multiple times
and avoids pathological cases where similar features get created repeatedly.

Zipties continue to grow new bundles indefinitely. If desired, you can set up an external
check and stop creating new bundles once you have as many as you want.

### Multiple layers

Because bundle activities are also valued between zero and one, the outputs
of one zipie can serve as the inputs to another. The bundles created
in one ziptie can serve as the cables, the inputs, to the next.

This allows for hierarchical clustering. Low level cables can be bound into bundles
in the first zipties, and these can then be bound into yet more
complex bundles in the next. This process can be repeated as many times as you like.
