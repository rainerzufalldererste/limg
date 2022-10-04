  <a href="https://github.com/rainerzufalldererste/limg"><img src="https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/image.png" alt="limg" style="width: 1024px; max-width: 100%"></a>
  <br>

## What is limg?
limg is an experimental (and very much work in progress) lossy raster image codec aimed towards compressing images of various origins (photos, graphics) pretty well. It's written in C++.

## How does it work?
limg tries to break up the image into blocks of similar linear factorization (so, basically two extreme point colors that all other colors in the block can be represented as a factor of).

For the image above, the following blocks are generated:  

![limg - Blocks](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/blocks.png)

The estimated linear factors of those blocks would then be these two colors:

| Factor A | Factor B |
:-------------------------:|:-------------------------:
![limg - Factor A](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/a.png)| ![limg - Factor B](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/b.png)

Now, every block is factorized into these colors resulting in a value between 0 and 1 (where 0 corresponds to completely being comprised of color A and 1 completely color B and 0.5 a 50/50 mix of both of them) for each pixel in a block.

![limg - Factorization](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/factors.png)

Since some of the linear factors are reasonably similar, we can get away with using less bits for the mix in some of the cases. Dithering is applied during the bit crushing process in order to reduce banding issues. (In the image below, lighter color represents less bits being required to represent those blocks to the desired accuracy)

![limg - BitCrush](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/bits.png)

The resulting image looks very similar to the original. The image at the top is already the decompressed version and features only very minor compression artifacts.

This is the difference to the original, boosted by a whole lot in order to make the errors visible. (The original basically looked like a completely black image.)

![limg - Difference](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/error.png)

For reference, this was the original image:

![limg - Original](https://raw.githubusercontent.com/rainerzufalldererste/limg/master/assets/original.png)