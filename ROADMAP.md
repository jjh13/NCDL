# NCDL Roadmap

This document outlines our plans and goals for [Your Project Name]. It includes both what we plan to work on in the near 
future and long-term plans. These may shift based on community needs and feedback. 
The ultimate goal of this project is to build a community around
practical non-Cartesian processing; reaching the ML community 
is the best way to achieve this goal.

## Short-Term Goals

These are the goals we're planning to work on in the next 3-6 months:

1. **Improve Testing**: We want to increase our code coverage and introduce performance tests. We have the inklings of 
   this in the paper, but we want an overall test framework that can be run release-to-release to track 
   progress and identify regressions.
2. **Better Documentation**: We're aiming to have better onboarding guides for new contributors. This is likely somewhat
   tricky, due to the obscure nature of the content.
3. **Bug Fixes**: We're currently prioritizing fixing known issues to provide a better user experience. It is ultimately 
   important that any bug
4. **API Cleanup**: Some functions/methods are not very consistent, or parameters may have confusing names. We need to 
   fix these issues rather critically.

## Long-Term Goals

These are the goals that we're planning to work on over the next 1-2 years:

1. **Scalability**: We want to ensure that NCDL can scale to meet larger demands; this may ultimately result in a 
   re-write of the core `LatticeTensor` object in C++. Additionally, we need specialized algorithms
   for convolution and pooling.
2. **New Features**: We plan to add new features as they're found useful, one big one is interpolation (i.e. think 
   grid_sample or interpolate). There's a huge body of work on this subject, and it's hard to get this exactly right 
   with the level of generality we would like.
3. **Improve Accessibility**: We would also like to make NCDL more accessible to users, including better conceptual 
   guidance (i.e. tutorials and documentation) and translations to other languages.

## Can I Help?

If you're interested in contributing to any of these goals (or if you have ideas for new ones), we would love to hear from you. Please check out our [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get started.

## Feedback

Your feedback keeps this project alive. If you have specific features you want to be added to the roadmap, please create an issue on our [Issues page](https://github.com/yourusername/yourrepositoryname/issues).

*Please note that this roadmap may change based on our understanding of what the community needs and our capacity to address those needs. Always stay updated with the latest version of the roadmap.*

