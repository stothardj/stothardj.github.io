---
layout: post
title:  "Generating a kids app using AI"
date:   2026-06-19 15:03:06 -0700
categories: ai
---

My daughter just turned three years old so she's allowed some screen time.
This could also be an opportunity to practice her fine motor skills, but the
first couple apps I checked out seemed a bit underhanded. They were trying to
get me into an annual subscription just for tracing some pictures.

The promise of AI is that I should be able to just ask an agent to make this
app and let my daugter try it. So let's try it.

## The Final Product

Let's skip ahead a bit. What did I actually generate in the end?

Well, there's the phone app of course. But let's also talk about the content.
How do we get images for the kid to trace?

### Step 1: Generate the list of images to generate

This is pretty straight forward. I just asked Gemini to pick 25 topics and 4 subjects for each topic.

### Step 2: Generate the images

Next was generating each of the images. So using Python I just looped over the list generated above I just called Gemini directly:

{% highlight python %}
prompt = f"""
Generate a cartoon image of a {subject}. Use the bright, vibrant, cute style that you might expect from a phone app icon. Keep it accurate. Only include the {subject}; no other objects or effects. Use a plain white background.
"""

interaction = client.interactions.create(
  model='gemini-3.1-flash-image',
  input=prompt,
)

full_path = dir_path / Path(f'{subject}.png')
full_path.write_bytes(base64.b64decode(interaction.output_image.data))
{% endhighlight %}

The only quirk here is I learned it's a lot better to ask for a "white background" and not "no background".
Occasionally "no background" leads to the checkerboard pattern which you often see representing transparency online.
But it's not transparent, it's actually Gemini mimicing the pattern. Quite funny, but it also points out a useful
lesson when working with these unpredictable genai tools. Try things on small samples and save intermediate steps so
mistakes can be re-run.

### Step 3: Pick the backgrounds

Each image is going to have to be placed on a background. You may be wondering why these are generated separately.

Well, in some other processing steps I'll want to only operate on the foreground and it's actually easier to combine
a forground image only a background image than it is to try to separate them after the fact.

Again I'm calling Gemini from python in a loop, but this time I want to output JSON instead of an image.

{% highlight python %}
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
{% endhighlight %}

As any AI expert will tell you, you really want to scream at it to output JSON.

### Step 4: Generate the background images

This is actually exactly as generating the background images.
I actually didn't ask Gemini to put the foreground image in place, but it might have worked.
Don't worry, we'll get to multimodal inputs soon.

{% highlight python %}
prompt = f'Generate {background.lower()}. Leave space in the middle of the image for me to add the subject in later.'
interaction = client.interactions.create(
    model='gemini-3.1-flash-image',
    input=prompt,
)
{% endhighlight %}


### Step 5: Highlighting the "important" parts of the image

I'm a patient adult but even I don't have the patience to trace every feather a peacock has.
But how can I define what the "important" parts of an image are. It's not just the biggest.
The peacock's feathres are bigger than the rest of it.

This lack of clear success criteria led me to again, just see what would happen if I asked
Gemini to choose the important parts. After a lot of trial and error I managed to create a prompt
which led to decent results. The trick was to require the highlights be overlayed onto the original
image. Without this Gemini had a strong tendancy to just generate a different image.

{% highlight python %}
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
{% endhighlight %}

### Step 6: Trace the image

So far all the images are pngs, but I really need a vector file so that the app knows where the
kid needs to trace. png -> svg is not trivial, and I'll talk about the failed paths later.
This is perhaps the most interesting part of the project.

First we import the image and convert it to black and white using OpenCV.
Each image will actually be generated with multiple thresholds for the greyscale to
black and white conversion. Unfortunately there was not one divider which was universally
best across images.

{% highlight python %}
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
{% endhighlight %}

Now that we have a black and white version of the image, it's time
for some libraries to do the processing. skelotonize from `skimage.morphology`
makes lines as thin as possible without breaking the path, but it
expects the inverted version of the black and white image so we
invert it using `skimage.util`. Next we analyze the skeleton using
`skan`. This sometimes fails within the library, so if that happens
we just hope that it works for a different threshold. `skan` generates
path coordinates, but it will generate many extra nodes. A straight line
will end up with a node at every step along the path. `rdp` fixes this
by simplifying the nodes in the path. It even has a configurable epsilon
fuzzy factor.

{% highlight python %}
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
{% endhighlight %}

### Step 7: Manually putting it together

There's a few things the AI doesn't yet do in this pipeline.

When paths intersect `skan` doesn't know which ones should connect to each other,
so it leaves all of them disjoint. I used Inkscape's node joining to do this.

`rdp` can simplify straight lines, but it doesn't simplify curves.
There's some work in this area I found, but nothing easily callable from python
that I have found yet. And as I already have each file open in Inkscape at the
last step, I just use "Simplify Path".

### Step 8: Importing the images into the app

The last step is importing the svgs into the app.
I did the first couple manually, then asked Antigravity CLI to generate a python
script to import the rest. It's not too intersting, the core of it looks like this:

{% highlight python %}
  page_svelte_content = f"""<script lang="ts">
import {{ onMount }} from 'svelte';
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
{% endhighlight %}

And with that I end up with a different page for each image to be traced. ex. `myapp/trace/panda`.
And it updates the JSON structure I'm using from the home page for the image picker.
