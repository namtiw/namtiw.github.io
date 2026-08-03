---
layout: post
title: It is just a beginning
subtitle: GSoC Wrap Up
thumbnail-img: /assets/img/gsoc.png
js: [/js/github-buttons.js, /js/required.js]
ext-js: https://ajax.googleapis.com/ajax/libs/jquery/1.12.0/jquery.min.js
css: [/css/required.css, /css/github-buttons.css]
tags: [GSoC, Terasology, Oreons]

---
This post serves as a central reference for all the work done during an highly eventful summer, to say the least. It has been a wonderful experience, as it involved working on an exciting gameplay scenario development and at the same time helped me learn a lot about how to write quality
code when dealing with a deadline and also that "upgradation" is not really a word :D.  
This post consists of links to all PRs opened and the work done during the summer. There have been a total of about *15000 lines of code* across multiple modules and the engine of the game. Not surprisingly this boosted my github stats to a whole new level.

<div align="center">
<figure>
<img src="/assets/img/contri.jpg">
</figure>
</div>

## Final Product
The final result of my 3 month contribution is a brand new gameplay scenario called the *Master Of Oreon*. This takes you to another world where you control creatures called the Oreons. There are three types of Oreons, the _King_, _Guard_ and the _Builder_. The player can 
 them work on tasks like building new structures, guarding the storage, planting crops, harvesting those crops etc.
All this work makes the Oreons lose health and develop hunger, now to replenish this health and satisfy their hunger the Oreons have to visit a Diner and Hospital. The Oreons can also train a myriad of attributes through various buildings like the Gym and Classroom, which they subtly 
request for if not present in the village. The player also needs to collect resources and put them into chests of buildings which requre them. For eg: the Diner requires a `Cookie Crop` block to be present in its chest in order to serve an Oreon. The player can also drop blocks inside the 
Storage building which adds them to the storage chest. This helps in the case where a missing block for a task can be fetched by the Oreons _automagically_ and subsequently perform the original task. A minimal viable product (MVP) was finished over the summer. 
 The module currently consists of three buildings -- _Diner_, _Laboratory_ and _Storage_ -- and one of them can be upgraded; and there is a single type of resource and a single _Research_  recipe.  However, adding new content is a simple matter of defining a prefab file for the _Books_ with a research
 recipe or defining a _StructureTemplate_ for a new building.

### Relevant resources
  
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/F5CGjjePnbA" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
  </div>
  
  
- #### List of PRs opened
Here is a [link](https://github.com/pulls?utf8=%E2%9C%93&q=is%3Apr+created%3A2018-05-05..2018-08-13+author%3Anamtiw) to a list of PRs opened during the GSoC work period.
<div class="collapsiblecontainer">
<div id="BTs" class="collapsibleheader">
<div align="center">
<div style="color:white">
+ Click here for the individual PRs opened
</div>
</div>
</div>
<div class="collapsiblecontent">
<b> MOO PRs </b><br>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/5"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/6"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/7"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/8"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/9"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/10"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/11"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/19"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/20"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/21"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/22"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/23"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/24"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/25"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/26"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/27"></div>
<div class="github-button" url="https://github.com/Terasology/MasterOfOreon/pull/28"></div>
<b>Engine PRs</b><br>
<div class="github-button" url="https://github.com/MovingBlocks/Terasology/pull/3410"></div>
<div class="github-button" url="https://github.com/MovingBlocks/Terasology/pull/3401"></div>
<b>Other Repositories</b><br>
<div class="github-button" url="https://github.com/Terasology/StructureTemplates/pull/27"></div>
<div class="github-button" url="https://github.com/Terasology/Oreons/pull/3"></div>
<div class="github-button" url="https://github.com/Terasology/Oreons/pull/4"></div>
<div class="github-button" url="https://github.com/Terasology/DynamicCities/pull/32"></div>
<div class="github-button" url="https://github.com/Terasology/CombatSystem/pull/35"></div>
<div class="github-button" url="https://github.com/Terasology/Books/pull/16"></div>
<div class="github-button" url="https://github.com/Terasology/ChangingBlocks/pull/3"></div>
<div class="github-button" url="https://github.com/Terasology/Behaviors/pull/14"></div>
</div>
</div>
- #### List of Blog Posts
  Click [here](/tags#GSoC) for a list of blog posts I made during the summer and [here](https://forum.terasology.org/threads/gsoc-2018-renovation-of-masteroforeon.2171/) for a thread of the project on Terasology's forum.
- #### Other Links
  - [Forum Thread](https://forum.terasology.org/threads/gsoc-2018-renovation-of-masteroforeon.2171/)
  - [Project Board](http://github.com/Terasology/projects/10)
  - [Project Proposal](https://drive.google.com/file/d/14q2m16WKgZCy5iVQbfsceRXeI7N8Wa5b/view)

## Future Prospects
The current implementation has some issues related to replicating the _queue_ of available tasks over the multiplayer network and similar serialization/deserilization quirks like invisible Oreons when starting a saved game. As mentioned earlier the module currently also lacks some
playable and interesting content like new types of tasks, resources to collect, buildings to construct, more upgraded versions of all buildings etc.
## Acknowledgements
I would like to thank all my mentors especially Mike 'mkienenb' Kienenberger, Rasmus 'cervator' Praestholm and Aresh 'vampcat' Mishra for consistently reviewing and  testing the PRs throughout the summer and pointing out how things should be done when working with a large 
codebase. I am looking forward to sticking around with this welcoming and helpful org post GSoC and contribute to the game overall especially polishing MOO's gameplay experience further.