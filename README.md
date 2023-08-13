  <a href="https://github.com/rainerzufalldererste/limg"><img src="https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/image.png" alt="limg" style="width: 1024px; max-width: 100%"></a>
  <br>

## What is limg?
limg is an experimental (and very much work in progress) lossy raster image codec aimed towards compressing images of various origins (photos, graphics) pretty well. It's written in C++.

## How does it work?
limg breaks the image up into blocks and calculates three linear factors the colors in this block consists of.

We then try to expand those blocks into blocks of similar color, and if possible, recalculate a better matching linear factor decomposition. For the image above we end up with the following factors:
![limg - Blocks](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/block.png)

For the image above, the first linear factor would be the colors between these two extremes:

| Minimum Factor A | Maximum Factor A |
:-------------------------:|:-------------------------:
![limg - Minimum Factor A](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/a_min.png)| ![limg - Maximum Factor A](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/a_max.png)

This already covers most of the colors of this block, but certainly not all, so the second linear factor is the most prevalent _direction_ of error from the first two extremes.
Since these directions are relative from the best color match from the first two extremes, they are displayed here in relation to 50% gray.

| Minimum Factor B | Maximum Factor B |
:-------------------------:|:-------------------------:
![limg - Minimum Factor B](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/b_min.png)| ![limg - Maximum Factor B](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/b_max.png)

In some cases even after this step there still remains some difference to the original color. The third linear factor works just like the second one to get us closer to the original colors.
Since this is still a relative color, we'll again display it in relation to 50% gray.

| Minimum Factor C | Maximum Factor C |
:-------------------------:|:-------------------------:
![limg - Minimum Factor C](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/c_min.png)| ![limg - Maximum Factor C](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/c_max.png)

Currently these factors aren't particularly optimal and just rely on the average error direction, which may not be a good idea in some cases.

Now, every block is factorized into these colors resulting in a value between 0 and 1 for each of the factors (where 0 corresponds to the minimum and 1 the maximum) for each pixel in a block.

| Factor A | Factor B | Factor C |
:-------------------------:|:-------------------------:|:-------------------------:
![limg - Factor A](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/fac_a.png)| ![limg - Factor B](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/fac_b.png)| ![limg - Factor C](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/fac_c.png)

Since some of the linear factors are reasonably similar, we can get away with using fewer bits for the mix in some of the cases. Dithering is applied during the bit crushing process in order to reduce banding issues. (In the image below, lighter color in each channel represents fewer bits being required to represent a blocks to the desired accuracy; Red = Factor A; Green = Factor B; Blue = Factor C)

![limg - BitCrush](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/bits.png)

The resulting image looks very similar to the original. The image at the top is already the decompressed version and features only very minor compression artifacts.

This is the difference to the original, boosted by a whole lot in order to make the errors visible. (The original basically looked like a completely black image.)

![limg - Difference](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/error.png)

For reference, this was the original image:

![limg - Original](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/original.png)
