---
layout: post
title:  "Generating a kids app using AI"
date:   2026-06-19 15:03:06 -0700
categories: ai
---

My daughter just turned three years old so she's allowed some screen time. This
could also be an opportunity to practice her fine motor skills. The first
couple apps I checked out felt like a ripoff though. They were trying to get me
into an annual subscription just for tracing some pictures.

The promise of AI should be that I can just ask an agent to make this app and
let my daugter play. So let's try it.

## The Final Product

Let's skip ahead. What did I actually generate in the end?

Before we talk about the phone app let's talk about the content. How do we get
images for the kid to trace?

### Step 1: Generate the list of images to generate

This is pretty straight forward. I just asked Gemini to pick 25 topics and 4
subjects for each topic.

### Step 2: Generate the images

![Fairy](/assets/images/fairy.png)

Next was generating each of the images. Using python I just looped over the
list generated above and had it just call Gemini using Google's genai library
for each subject in the list. Here's what was in that loop:

```python
{% raw %}
prompt = f"""
Generate a cartoon image of a {subject}. Use the bright, vibrant, cute style that you might expect from a phone app icon. Keep it accurate. Only include the {subject}; no other objects or effects. Use a plain white background.
"""

interaction = client.interactions.create(
  model='gemini-3.1-flash-image',
  input=prompt,
)

full_path = dir_path / Path(f'{subject}.png')
full_path.write_bytes(base64.b64decode(interaction.output_image.data))
{% endraw %}
```

The prompts throughout this post will reflect a lot of the trial and error that
comes with trying to get genai to do the right thing. One quirk I found here is
it's a lot better to ask for a "white background" than "no background".
Occasionally "no background" leads to the checkerboard pattern which you often
see representing transparency online. But it's not transparent, it's actually
Gemini mimicing the pattern.

### Step 3: Pick the backgrounds

Each image is going to have to be placed on a background, so the next step is
choosing what those backgrounds will be. I do this independently of actually
generating the background images so I can quickly take a look at what it's
coming up with. Generating images is cheap, but not free.

You may be wondering why the backgrounds are generated separately. Well, in
some other processing steps I'll want to only operate on the foreground and
it's easier to combine a forground image with a background image than it is to
try to separate the foreground from an image which generated both at the same
time.

Again I'm calling Gemini from python in a loop, but this time I want to output
JSON instead of an image.

```python
{% raw %}
        prompt = f"""There's a cartoon image of a {subject} which I need to put on top of a background for a kid's app. What is an appropriate background to use for this?

IMPORTANT: Respond using JSON! The response to this prompt will be used non-interactively. Format as follows:

{{"background": "Your background choice"}}
        """
response = client.models.generate_content(
    model='gemini-3.5-flash',
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type='application/json',
        response_schema=Background,
    ),
)

background = Background.model_validate_json(response.text)
background_choice[topic].append((subject, background.background))
{% endraw %}
```

As any AI expert will tell you, screaming at models that they should output
JSON is a critical part of the job.

### Step 4: Generate the background images

![Fairy Background](/assets/images/fairy-background.png)

This is actually exactly as generating the background images. I actually didn't
ask Gemini to put the foreground image in place, but it might have worked.

```python
{% raw %}
prompt = f'Generate {background.lower()}. Leave space in the middle of the image for me to add the subject in later.'
interaction = client.interactions.create(
    model='gemini-3.1-flash-image',
    input=prompt,
)
{% endraw %}
```

I didn't include the foreground image in the prompt as I was concerned it would
influence the background image in unexpected ways. However this also led to
some surprises. Sometimes the background would include elements that made the
foreground not make as much sense. For example the foreground image being a
drumset and the background being a rockband stage full of instruments...
including another drumset. Other times the generated background made it clear
that the background description had lost crucial detail. For example some of
them were realisitc instead of being cartoons! If I were to revisit this
pipeline I would probably either merge the steps of choosing and generating the
backgrounds or include the foreground alongside the chosen background
description.

### Step 5: Highlighting the "important" parts of the image

![Fairy Simplified](/assets/images/fairy-simplified.png)

I'm a patient adult but even I don't have the patience to trace every feather of a
peacock.

But how can I define what the "important" parts of an image are. It's not just
the biggest. The peacock's feathers are bigger than the beak, but it makes more
sense for the child to practice tracing the beak.

This lack of clear success criteria led me to just see what would happen if I
asked Gemini to choose the important parts. After a lot of trial and error I
managed to create a prompt which led to decent results. The trick was to
require the bold lines be overlayed onto the original image. Without this,
Gemini had a strong tendancy to just generate a different image.

```python
{% raw %}
TEXT_PROMPT = f"""
Create a version of this image with BOLD well-defined lines a kid should trace in a tracing app. Only include the very important lines in order to get a basic outline of the form. The toddler will only have patience for at most 10 lines. Use simple curves for each line, ignoring small bumps and details.

IMPORTANT: The outline MUST overlay onto a low opacity copy of the original image. Keep the original image present.

DO NOT add any other elements to the image. Just increase the thickness of lines to be traced.
"""

def process_image(input_path, rel_path):
    subject = rel_path.stem
    output_name = f"{rel_path.stem}-simplified.png"
    output_path = generated_bold / rel_path.parent / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Processing {input_path} (subject={subject}) -> {output_path}")

    image = Image.open(input_path)
    response = client.models.generate_content(
        model='gemini-3.1-flash-image',
        contents=[
            TEXT_PROMPT,
            image,
        ],
        config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
        )
    )

    for part in response.parts:
        if part.text is not None:
            print(f'From model output: {part.text}')
        elif part.inline_data is not None:
            print('Has inline data!')
            image = part.as_image()
            image.save(output_path)
{% endraw %}
```

### Step 6: Trace the image

![Fairy Traced](/assets/images/fairy-traced.svg)

So far all the images are pngs, but I really need a vector file so that the app
knows where the kid needs to trace. Converting a png to an svg is not trivial,
and I'll talk about the failed paths later. This is perhaps the most
interesting part of the project.

First we import the image and convert it to black and white using OpenCV. Each
image will be generated with multiple thresholds for the greyscale to black and
white conversion. Unfortunately there was not one divider which was universally
best across images.

This step generally had the property that the semi-transparent overlay
disappeared. There are some images which are currently excluded from the app
because they didn't auto-trace well. The plan is to try converting them to HSV
and then dropping things with a low saturation. I'll have to experiment though.

```python
{% raw %}
# Read the image using cv2
img = cv2.imread(str(input_path))
if img is None:
    print(f"Error: Could not read image '{input_path}'", file=sys.stderr)
    sys.exit(1)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

height, width, _ = img.shape

for thresh in thresholds:
    output_name = f"{rel_path.stem}-{thresh}.svg"
    output_path = generated_autotrace / rel_path.parent / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {input_path} (threshold={thresh}) -> {output_path}")

    # Convert to black and white
    bw = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)[1]
{% endraw %}
```

Now that we have a black and white version of the image, it's time for some
libraries to do the processing. skelotonize from `skimage.morphology` makes
lines as thin as possible without breaking the path, but it expects the
inverted version of the black and white image so we invert it using
`skimage.util`. Next we analyze the skeleton using `skan`. This sometimes fails
within the library, so if that happens we just hope that it works for a
different threshold. `skan` generates path coordinates, but it will generate
many extra nodes. A straight line will end up with a node at every step along
the path. `rdp` fixes this by eliminating the redundant nodes in the path. It
even has a configurable epsilon fuzzy factor.

Many of these libraries actually were suggested by Gemini. The trick is that it
also suggested many idea which don't work, so I had to visualize where each
suggestion was taking the image.

```python
{% raw %}
# Invert black and white
bw_invert = invert(bw)

# Make sure very thin
skeleton = skeletonize(bw_invert)

# Analyze and simplify the skeleton.
try:
    # Calling "Skeleton" sometimes fails
    skel = Skeleton(skeleton)
    simplified_paths = [rdp(skel.path_coordinates(i), epsilon=0.5) for i in range(skel.n_paths)]

    # Write paths to svg.
    write_skeleton_to_svg(simplified_paths, width, height, output_path)
except Exception as e:
    print(f'Failed to skeletonize {output_name}: {e}')
{% endraw %}
```

### Step 7: Manually putting it together

![Fairy Final](/assets/images/fairy.svg)

There's a few things the AI doesn't yet do in this pipeline.

When paths intersect `skan` doesn't know which ones should connect to each other,
so it leaves all of them disjoint. I used Inkscape's node joining to do this.

`rdp` can simplify straight lines, but it doesn't simplify curves.
There's some work in this area I found, but nothing easily callable from python
that I have found yet. And as I already have each file open in Inkscape at the
last step, I just use "Simplify Path". Note, python can actually call Inkscape but
that's only intersting if I solve automating the path joining.

### Step 8: Importing the images into the app

The last step is importing the svgs into the app.
I did the first couple manually, then asked Antigravity CLI to generate a python
script to import the rest. It's not too intersting, the core of it looks like this:

```python
{% raw %}
  page_svelte_content = f"""<script lang="ts">
import DrawArea from '$lib/components/DrawArea.svelte';
import {pascal_name}Image from './{component_filename}';

const {paths_var_name} = [
{paths_list_str}
];
</script>

<DrawArea
  paths={{{paths_var_name}}}
  Background={{{pascal_name}Image}}
  width={w_str}
  height={h_str} />
"""
  with open(os.path.join(dest_dir, "+page.svelte"), 'w', encoding='utf-8') as page_f:
    page_f.write(page_svelte_content)
{% endraw %}
```

And with that I end up with a different page for each image to be traced. ex.
`myapp/trace/panda`. And it updates the JSON structure I'm using from the home
page for the image picker. I could have made it all data in a JSON file, but I
find it easier to just have one file per image.

## The phone app

### The first version

The first version of the app was written by Gemini. I didn't know how it worked.

And I really really did not like that.

When I had people test it I'd have to ask Gemini to make it easier or harder.
To make it less precise about how far the trace had to be to the line or the percent
of the line that had to be covered. It would make the tweaks but the app never
felt right.

### The second version

![Tiger Trace Home](/assets/images/tiger-trace-home.png)

So I rewrote it myself.

There's not much to say about the stack. It's the one written [about by someone
else in this
blog](https://khromov.se/how-i-published-a-gratitude-journaling-app-for-ios-and-android-using-sveltekit-and-capacitor/).

There's a Svelte website which is wrapped by Capacitor JS in a web view to
launch as an app. Drawing is just accomplished by adding a dynamic polyline to
the svg.

Then everything else is handled by libraries:
 - Svgs paths are parsed by `svg-path-parser`.
 - Zoom animations are controlled by `gsap`.
 - Colors are from `js-colormap`.

I could have taken a more guided approach to getting an agent to build it, but
I was curious to learn Svelte anyway. I did get LLM help with a couple issues,
especially when it only reproduced on actual phone hardware.
 
But with a fairly simple setup with a few libraries, you can draw.

![Tiger Trace Drawing](/assets/images/tiger-trace-drawing.png)

## Generating images: All the paths which didn't work

An LLM could not do this by itself. That was my first attempt. Just ask an
agent to build this app. It didn't work. But LLMs did allow exploring lots of
bad ideas faster.

### Failure 1: Have the coding agent do everything

Asking Antigravity to build the app from scratch including generating all media
went better than I'd have expected. You could trace and it even make a somewhat
competent cat consisting of a circle for the head and two triangles for the
ears by manually writing an svg file.

But that was about as complicated as the images could get, and many of them
were quite broken. After exhausting my freebie Antigravity quota I tried Claude
with playwrite to try to fix the images. I told Claude it was fixing the work
of a different agent and it took the time to criticize Gemini's work, before
making the images even worse.

Clearly asking coding agent's to generate images is just the wrong task for the
job.

### Failure 2: Asking the AI agents to write an import pipeline

Ok so Nanobana can generate the images but how can I make these into svgs.

Inkscape has a "Trace bitmap" feature which even supports centerline tracing.
It's not bad but everything is put into one line. There's "break paths" but it
does not break into anything resembling natural tracing. Many things remain
connected to the wrong thing.

I tried going down this path where I'd ask Nanobana to make each line with a
unique color. The idea was I could mask the image to look at one line at a
time. This only kind of works. There's actually lots of noise in AI generated
images and even if it's not visible to us it really messes up attempts at color
masking. Nanobana is also not very good at making strictly unique colored
lines. It reuses colors, and it ends up blending colors when lines come close
to each other. The noise from this set of images just foiled any attempts at
post-processing. I'm still convinced this path could work if I spent more time
and actually had a classifier try to group pixels based on the noisy lines
coming from Nanobana, but it was going to take more time to do this than the
rest of the project.

When I asked an AI agent to write a python script for importing the images it
made something which just output images which were clearly broken. No amount of
feeding the broken images into the agent and asking it to fix things was going
to get anywhere.

### Failure 3: Direct text to SVG

What if Nanobana is introducing too much complexity?

Gemini convinced me to try getting [diffvg](https://github.com/BachiLi/diffvg)
working to directly generate vector graphics. And while the project looks cool,
getting it working on RunPod was taking too long as I was teaching myself
Docker just to get it to work. Considering it's a 6 year old project, and AI
has moved a lot in the past few years, I decided it probably wasn't work
persuing.

Could be wrong though, I just never got it working.

### What got me out of the failure cycle

Ultimately what got me out of the cycle of failure was putting down the
agents and visualizing things one step at a time.

I opened up colab and visualized how I was transforming an image step-by-step
as I ran it through various OpenCV, skeletonize, and other libraries.

![Pirate Ship](/assets/images/pirate-colab.png)

## Conclusion

All of this struggle was probably a pretty average experience with AI.

LLMs can do some very well things. It's honestly surprising how well it picks out the "important"
lines with no further direction.

Other times it was fairly frustrating. I didn't expect to need to provide so
much manual work on building a pipeline for creating the tracing guides.

How people view the output of this project will depend on their view of AI. Is
it AI slop? The images are clearly the work of AI image gen. The code had some
AI influence. Or it's the promise of a dad being able to generate an app for
their kid given a very limited amount of free time.

If you have an Android phone you'll be able to download TigerTrace when it
launches soon. Maybe iPhone later if I decide it's worth paying $99/year to
give something away for free on that app store too.

Or generate your own app. That is the promise of AI after all.
