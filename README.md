# SeaMeInIt
SMII(gol?)  Parametric design of next-generation, made-to-fit clothing. We aim to address major factors such as climate adaptation and refugees, accessiblity, as well as hobbyists and cosplayers.

Please see our docs:
- [We use Ben Afflec as our test fixture because we thought he'd appreciate an Iron Man suit of his own.](docs/afflec_image_preprocessing.md)
- [Avatar Model](docs/avatar_model.md)
- [Engine Integration](docs/engine_integration.md)
- [Hard Layer Export](docs/hard_layer_export.md)
- [Hard Layer Shell](docs/hard_layer_shell.md)
- [Hard Shell Clearance](docs/hard_shell_clearance.md)
- [Measurement Inference](docs/measurement_inference.md)
- [ROM Latent Space Notes](CONTEXT.md) (see lines 1-120 for the validity and latent-space framing)
- [Schemas](docs/schemas.md)
- [Undersuit Generation](docs/undersuit_generation.md) - Currently in progress.
- [Cooling](docs/modules/cooling.md)
- [Deployable Tent](docs/modules/tent.md)
- [Agents](AGENTS.md)
- [Contributing](CONTRIBUTING.md)
- [Roadmap](ROADMAP.md)
- [TODOs](TODO.md) (see `CONTEXT.md` lines 1141-1147 for TODO hygiene guidance)

## Current focus: manufacturable panels

The undersuit pipeline needs an explicit "panel" layer between geometry and export. We should treat paneling and sewability as
first-class constraints before flattening, then run deterministic boundary regularization so outputs become clean, vector-ready
patterns.

Near-term work:
- Define a Panel abstraction (3D patch + 3D/2D boundaries + seam partners + grain direction + distortion/sewability budgets).
- Enforce sewability constraints before LSCM/ABF (split panels when thresholds are exceeded).
- Add boundary regularization stages (resample, clamp curvature/turning, suppress tiny features, spline fit) with structured issues and split suggestions.

## 🛡️ License Summary

SeaMeInIt 🛠️ is currently licensed under GPL-3.0, with intention towards a custom [Business Source License](./LICENSE) or similar.

> Commercial use is allowed under the same terms (with copyleft).
> We may migrate to a dual-license model in the future to better align with our ethical and sustainability goals.


- ✅ Free for personal, nonprofit, educational, and humanitarian use
- ✅ Source available for all
- ❌ Commercial use **only if your org makes under $500K USD/year**
- 💬 Contact us for commercial licensing if you're above that threshold
- 🔓 After 3 years, each version becomes AGPL-3.0 open source

This helps keep our work open, sustainable, and ethically aligned.

## 🧊 Avatar body assets and tooling

SeaMeInIt understands multiple SMPL-family asset bundles. Pick the bundle that matches your licensing constraints, then invoke
the provisioning helper to download, verify, and extract the archive into `assets/<model>/`.

| Bundle | Command | License | Notes |
| ------ | ------- | ------- | ----- |
| `smplx` | `python tools/download_smplx.py --model smplx --dest assets/smplx` | [SMPL-X model license](https://smpl-x.is.tue.mpg.de/) | Requires an authenticated download URL or cookie session. |
| `smplerx` | `python tools/download_smplx.py --model smplerx --dest assets/smplerx` | [S-Lab License 1.0](https://raw.githubusercontent.com/caizhongang/SMPLer-X/main/LICENSE) | Open download; redistribution limited to non-commercial use. |

Provisioning flow:

1. Install the tooling extras: `pip install -e .[tools]`
2. For licensed downloads export the authenticated link via `SMPLX_DOWNLOAD_URL` and optional `SMPLX_AUTH_TOKEN` cookie.
   Override the canonical URLs with `--url` or reuse a local archive with `--archive`.
3. (Optional) Supply `--sha256 <checksum>` to enforce integrity verification. Checksums can be recorded per manifest for future audits.
4. Run the helper using one of the commands above. The script writes a `manifest.json` summarising the bundle (source URL,
   license, checksum, top-level contents) so runtime tooling can automatically discover the assets.

Inspect the manifest at any time to confirm the bundle metadata, for example: `jq '.license, .sha256' assets/smplerx/manifest.json`.

### Visualising fitted meshes

The Afflec demo exists purely as a smoke test that runs our pipelines against a few tongue-in-cheek stills of Ben Afflec. We do **not** ship or rely on an "Afflec model"—it's simply a lightweight fixture bundle that lets contributors exercise the CLI end to end without redistributing sensitive body scans. The demo writes a NumPy archive containing the fitted SMPL-X body mesh to `outputs/afflec_demo/afflec_body.npz`. Install the viewer dependencies with `pip install trimesh "pyglet<2" scipy`, then launch an interactive preview:

```bash
python tools/view_mesh.py outputs/afflec_demo/afflec_body.npz
```

Pass `--info-only` to print mesh statistics without opening the window, or `--process` to let Trimesh repair normals and merge duplicate vertices before viewing.

# END TECHNICAL INFO


# **Next-Generation Adaptive Suit Platform** - Longform Vision and Roadmap

## Introduction and Vision

Our project aims to develop **adaptive, high-performance protective suits** that bridge the gap between niche professional gear and everyday consumer apparel. We envision multi-purpose suits that can serve astronauts, firefighters, divers, and even climate refugees - and eventually be affordable and accessible to the general public. Crucially, this initiative is guided by ethical principles: we plan to **openly share knowledge and designs** (in an open-source spirit) while maintaining a sustainable business model for longevity. The end goal is a suit platform that is **lightly profitable but largely open and human-centered**, delivering life-saving functionality without prohibitive costs. We prioritize humanitarian impact (e.g. aiding first responders and displaced populations) and **environmental sustainability**, striving for materials and production methods that reduce waste. In the long term, we want these suits to be as **ubiquitous and affordable as emergency tents**, enabling mass deployment in disaster relief and climate crises.

## Target Markets and Use Cases

**1\. Specialized Sectors (Today):** We are initially targeting high-need professional domains where advanced protective suits are already critical. These include:  
\- **Space Exploration:** Space agencies and aerospace firms currently invest heavily in bespoke spacesuits for astronauts. Modern EVA suits cost millions, but the space sector is evolving - with **new materials research** (e.g. ESA's **PEXTEX project** developing textiles for lunar suits) promising more robust and eventually more cost-efficient gear[\[1\]](https://comex.fr/en/news-en/new-intelligent-materials-for-future-space-suits/#:~:text=The%20objective%20of%20the%20project,and%20the%20Austrians%20from%20OeWF)[\[2\]](https://comex.fr/en/news-en/new-intelligent-materials-for-future-space-suits/#:~:text=identify%20materials%20capable%20of%20resisting,the%20many%20external%20aggressions). There's also a growing **DIY and democratized spaceflight movement**; for example, a researcher built a functional pressure suit for under \\\$30k (versus NASA's multi-million suits), illustrating that low-cost innovation is possible[\[3\]](https://medium.com/the-journal-of-critical-space-studies/pacific-spaceflight-ab25c5b4347c#:~:text=Unlike%20Dolgov%20and%20other%20pressure,NASA%20or%20private%20aerospace%20companies). As space tourism and lunar missions expand, there will be a market for **more accessible space-grade suits**. Our platform positions us to serve this emerging need with open innovation, ensuring **space doesn't remain the exclusive domain of the military-industrial complex** but benefits humanity broadly[\[4\]](https://medium.com/the-journal-of-critical-space-studies/pacific-spaceflight-ab25c5b4347c#:~:text=falling%20out).

- **Firefighting and First Responders:** Firefighters face extreme heat and hazardous conditions; state-of-the-art turnout gear now integrates advanced cooling and filtration. For instance, "personal cooling systems" for firefighters and soldiers come in **liquid-cooled, air-cooled, or phase-change** variants[\[5\]](https://oceanit.com/products/active-cooling-suit/#:~:text=Current%20PCSs%20fall%20into%20three,cooled%2C%20and%20phase%20change%20vests). Oceanit's recent **Super Cool Vest** exemplifies innovation here - it can circulate coolant through novel polymer tubes, yielding ~50% better thermal conductivity than standard materials and outperforming existing cooling vests by 30% in heat extraction[\[6\]](https://oceanit.com/products/active-cooling-suit/#:~:text=Current%20PCSs%20fall%20into%20three,cooled%2C%20and%20phase%20change%20vests)[\[7\]](https://oceanit.com/products/active-cooling-suit/#:~:text=A%20key%20shortcoming%20of%20other,conductivity%20than%20PVC%20or%20Tygon). This indicates a **trend toward active cooling wearables** for extreme-heat occupations. Our suit will build on such advances, potentially incorporating **liquid cooling garments (LCGs)** or phase-change elements to protect users like firefighters from heat stress[\[8\]](https://oceanit.com/products/active-cooling-suit/#:~:text=Oceanit%20is%20developing%20advanced%20Liquid,operating%20in%20harsh%20thermal%20environments)[\[9\]](https://oceanit.com/products/active-cooling-suit/#:~:text=Oceanit%E2%80%99s%20%E2%80%9CSuper%20Cool%20Vest%E2%80%9D%20was,by%20using%20four%20key%20innovations). Additionally, first responder suits may include **respiratory protection and sensors** in the future. Our platform will cater to these professionals with modular add-ons (e.g. cooling packs, air filters) while driving down cost for wider deployment (think volunteer firefighters or communities in wildfire zones).
- **Diving and Marine Work:** Divers and underwater rescuers need protection from cold and pressure. **Heated dive suits** are an emerging innovation - modern wetsuits can include **battery-powered heating elements** to keep divers warm, extend dive times, and even assist those with medical cold sensitivity[\[10\]](https://blog.wetsuitwearhouse.com/what-are-heated-wetsuits/#:~:text=Quadri%20Abdur%20%C2%A0%C2%A0%20February%2018th%2C,2025%C2%A0%C2%A0%20Posted%20In%3A%20%20161)[\[11\]](https://blog.wetsuitwearhouse.com/what-are-heated-wetsuits/#:~:text=Heated%20wetsuits%20are%20specially%20designed,comfortable%20temperature%20inside%20the%20suit). Such technology shows how active thermal regulation is becoming feasible in wearable form. Our suit designs will explore **active heating modules** (for cold-water or high-altitude use) alongside insulation. We also consider new materials for waterproofing and flexibility under pressure, taking cues from high-end dive wetsuits and drysuits.
- **Disaster Relief and Climate Adaptation:** A driving motivation for this project is to support **humanitarian applications**. We are inspired by innovations like the **Sheltersuit**, a free distributed jacket that converts into a sleeping bag, made for homeless individuals and refugees using upcycled tent fabric and recycled sleeping bags[\[12\]](https://sheltersuit.com/en-us/#:~:text=The%20Sheltersuit%20is%20a%20wind,bag%2C%20and%20a%20duffel%20bag)[\[13\]](https://sheltersuit.com/en-us/#:~:text=The%20products%20we%20provide%20are,upcycled%20materials%20to%20reduce%20waste). Similarly, designers have prototyped a **jacket that unfolds into a tent**, made from Tyvek (a cheap, waterproof, recyclable material) to shelter refugees on the move[\[14\]](https://www.fastcompany.com/3055886/this-jacket-designed-for-refugees-transforms-into-a-tent#:~:text=The%20jacket%20is%20made%20from,like%20fabric%20used%20in%20envelopes)[\[15\]](https://www.fastcompany.com/3055886/this-jacket-designed-for-refugees-transforms-into-a-tent#:~:text=Image). These examples highlight the potential of multi-purpose, low-cost wearable shelters. Our long-term vision is to make protective suits that can double as emergency shelter, or provide cooling/warming as needed, to assist **climate refugees and disaster victims**. Such suits should be **ultra-affordable, durable, and made from fully recyclable or natural materials**, so they can be produced and deployed en masse like "hurricane tents" - but with the wearable convenience of clothing. The **market-transition strategy** is to start by serving specialized markets (space, military, firefighting) where budgets allow for R&D, then **scale up production to drive costs down**, and finally transfer the technology to humanitarian use at minimal cost. By planning for mass manufacturing and simplicity from the start (for example, using materials like Tyvek or modular components), we aim to achieve economies of scale that make humanitarian units **extremely cheap or free** for those in need, without sacrificing quality.

**2\. General Consumers (Future):** In the long run, we see a future where advanced protective wear isn't just for professionals. Climate change is making extreme conditions (heatwaves, wildfires, storms) more common, and outdoor enthusiasts or even everyday people may seek better protective clothing. The concept of **"climate-adaptive apparel"** is already entering the public consciousness[\[16\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=A%20jacket%20is%20fireproof%20and,removable%20side%20that%20filters%20water)[\[17\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=But%20after%20the%20creative%20process,%E2%80%9D). For example, designers have envisioned **fireproof, smoke-filtering jackets, and solar-powered gear** for an apocalyptic climate scenario[\[18\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=This%20new%20clothing%20line%20is,life%20after%20apocalyptic%20climate%20change)[\[16\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=A%20jacket%20is%20fireproof%20and,removable%20side%20that%20filters%20water) - and noted that many of these items are relevant _today_, not just in the future[\[19\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=But%20after%20the%20creative%20process,%E2%80%9D). Our project's end-state could tap into this consumer market by offering **lightweight versions of our suits** for personal use - imagine a jacket that keeps you cool during a heatwave and has built-in air filtration for smoke, or a modular suit for hikers that provides rain protection, cooling, and emergency shelter. Achieving general consumer adoption will require hitting retail-friendly price points and style/usability, which we believe is possible once the tech matures and scales. Ethically, we will balance this commercial aspect with our core humanitarian mission: for instance, profits from consumer sales could subsidize giving suits to firefighters in developing regions or refugees. By maintaining an **open innovation ecosystem** around the suit platform, we also invite hobbyists and makers to contribute - much like how open-source software thrives - which can accelerate improvements and spread adoption. Ultimately, **general consumers represent the broadest market to create impact**, and reaching them would mean our suits truly are as commonplace as everyday apparel, fulfilling our vision of mainstream, life-saving wearables.

## Ethical and Sustainable Design Approach

**Open Collaboration:** Our strategy is to position the project between full open-source and a traditional startup. The **core designs and software** will be open for transparency and community contribution, ensuring **no single entity "locks up" life-saving tech for profit**. However, we will implement a sustainable revenue model (e.g. selling physical suits or enterprise support) to fund R&D and maintain quality. This approach echoes the ethos that "space (and by extension advanced technology) belongs to humanity, not only to corporations"[\[4\]](https://medium.com/the-journal-of-critical-space-studies/pacific-spaceflight-ab25c5b4347c#:~:text=falling%20out). We intend to publish design files, material research, and even allow third-party manufacturing, while possibly trademarking a brand for quality assurance. Ethics are central - we will avoid exploitative labor in manufacturing, enforce supply chain transparency, and prioritize user safety over margins.

**Sustainable Materials:** Reducing waste and using **recyclable or natural materials** is a core requirement for our suits. The textile industry is moving toward sustainable tech, even in PPE. For instance, workwear fabric makers are **swapping virgin fibers for recycled ones** like recycled polyester (e.g. REPREVE® fibers from plastic bottles) and exploring renewable fibers like TENCEL™ Lyocell (wood pulp fiber with low water use)[\[20\]](https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear#:~:text=There%20are%20four%20essential%20elements,sustainable%20protective%20clothing%20supply%20chain)[\[21\]](https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear#:~:text=REPREVE%C2%AE%20recycled%20polyester%20fibres%20in,remaining%20durable%20during%20industrial%20laundering). These materials can maintain durability through industrial use and washing - crucial for reusable protective gear[\[22\]](https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear#:~:text=,compares%20favourably%20against%20virgin%20polyester)[\[23\]](https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear#:~:text=recycled%20plastic%20bottles%2C%20REPREVE%C2%AE%20fibres,remaining%20durable%20during%20industrial%20laundering). We will follow this SOTA closely, selecting fabrics that combine performance with sustainability. Examples include: organic or recycled cotton blends for comfort layers, bio-based or recycled **aramids** for fire/heat shielding (recycled aramid fibers are just emerging for protective clothing)[\[24\]](https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear#:~:text=Well,evidence%20of%20the%20product%27s%20origins)[\[25\]](https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear#:~:text=Recycled%20aramids%20are%20already%20regenerated,in%20the%20protective%20clothing%20industry), and recyclable polymers for structural components. We will design for **circularity** - meaning the suit components should be upcyclable or recyclable at end-of-life. In practice, that could mean using single-polymer layers where possible (to ease recycling), avoiding toxic treatments, and partnering with programs for take-back and recycling of used suits. The **Sheltersuit** model already shows how upcycling waste materials into functional gear can succeed at scale (each Sheltersuit uses discarded sleeping bags and tent fabric, simultaneously helping people and cutting waste)[\[26\]](https://sheltersuit.com/en-us/#:~:text=The%20outer%20shell%20is%20made,and%20contains%20an%20integrated%20scarf)[\[27\]](https://sheltersuit.com/en-us/#:~:text=labor%20force%20and%20made%20out,upcycled%20materials%20to%20reduce%20waste). Likewise, the refugee jacket-tent prototype chose Tyvek for being **fully recyclable and cheap** while durable enough for an entire journey[\[14\]](https://www.fastcompany.com/3055886/this-jacket-designed-for-refugees-transforms-into-a-tent#:~:text=The%20jacket%20is%20made%20from,like%20fabric%20used%20in%20envelopes)[\[15\]](https://www.fastcompany.com/3055886/this-jacket-designed-for-refugees-transforms-into-a-tent#:~:text=Image). Inspired by these, our deliverables will include a **materials R&D report** identifying natural or recycled alternatives for every major component of the suit. The ultimate goal is a suit that not only protects lives but also respects the planet - proving that high-tech protective gear _can_ be eco-friendly.

**Minimizing Clothing Waste:** By creating durable, multi-purpose suits, we aim to combat the throwaway culture of clothing. Each of our suits could replace multiple single-purpose items (coat, tent, cooling vest, etc.), thus reducing overall consumption. Additionally, by making suits "as cheap as hurricane tents," we envision scenarios like humanitarian agencies distributing our suits instead of piles of blankets, tarps, and garments that often end up discarded. Durability and repairability will be designed in - perhaps through **modular components** (for example, a damaged sleeve or cooling unit can be swapped out without trashing the whole suit). We will also explore **biodegradable coatings and fibers**, so if a suit does get thrown away, it leaves minimal trace. All these efforts position our project at the forefront of ethical, sustainable innovation in wearable technology.

## Adaptive Design Platform & AI Integration

A key differentiator for our project is the creation of a **digital suit design platform** that allows rapid prototyping and customization of suit models. Instead of traditional static CAD design alone, we're building a **programmatically editable 3D suit model** - essentially a parametric digital twin of the suit that can be modified via software or even natural language commands. This platform will be the foundation for iterative design and user-specific tailoring.

**Parametric Suit Model:** We will develop a detailed 3D model of the suit (initially a baseline design for an adult human) using parametric CAD techniques. This means key dimensions (height, limb length, joint circumferences, etc.) and features (pocket placements, etc.) are parameterized. Using a parametric/B-rep model allows us to easily adjust the geometry for different body sizes and to add or remove features in a controlled way[\[28\]](https://arxiv.org/html/2409.17106v1#:~:text=Computer,iteratively%20refine%20the%20final%20models)[\[29\]](https://arxiv.org/html/2409.17106v1#:~:text=Despite%20their%20capabilities%2C%20modern%20CAD,parametric%20CAD%20generation%2C%20making%20it). Modern CAD tools support constructing models via **history trees** of sketches and extrusions - our model will leverage this so that **textual or scripted instructions can alter early parameters and regenerate the final suit**.

**AI-Assisted Design (Text to CAD):** We are integrating state-of-the-art **Generative AI** to enable non-experts (or even an AI agent) to modify and create suit designs through simple text prompts. Notably, recent breakthroughs like **Text2CAD (2024)** demonstrate that it's now possible to generate _parametric CAD models_ from natural language descriptions[\[30\]](https://arxiv.org/html/2409.17106v1#:~:text=Prototyping%20complex%20computer,draw%20two%20circles%20with)[\[31\]](https://arxiv.org/html/2409.17106v1#:~:text=Currently%2C%20there%20are%20works%20on,dagger%20%2041%20https%3A%2F%2Fgithub.com%2FKittyCAD%2Fmodeling). Unlike earlier text-to-3D methods that produced uneditable meshes, Text2CAD yields genuine CAD construction sequences that a designer can later tweak[\[32\]](https://arxiv.org/html/2409.17106v1#:~:text=Currently%2C%20there%20are%20works%20on,body%2C%20and). In parallel, startups like **Zoo Design Studio** have introduced text-driven CAD editing in commercial tools - users can literally type something like _"add a 50mm diameter vent on the back"_ and the software will update the model accordingly[\[33\]](https://zoo.dev/text-to-cad#:~:text=Turning%20thoughts%20into%20complex%20mechanical,designs)[\[34\]](https://zoo.dev/text-to-cad#:~:text=After). These developments align perfectly with our needs. We plan to incorporate an **AI design assistant** that can interpret user requests (potentially through a fine-tuned large language model) and execute changes to the suit model. For example, a firefighter could request "make the suit's outer layer fireproof and add cooling tubing in the torso," and our system would respond by selecting a fireproof material setting and integrating a predefined cooling-tube subassembly into the CAD model. Under the hood, this could be achieved by having the AI generate a script or high-level CAD instructions (akin to how OpenAI's Codex can generate code). Indeed, researchers have proposed frameworks where **LLMs generate CAD API scripts (OpenSCAD, etc.) from text prompts**[\[35\]](https://medium.com/@gwrx2005/ai-driven-design-tool-for-blueprints-and-3d-models-a9145f5ee537#:~:text=AI,3D%20model%20or%20drawing). We will leverage such approaches: the AI might output a Python script for FreeCAD or a custom "KittyCAD" script (Zoo's backend language) to modify the base suit model. The user or design team can then review and fine-tune the result. This **prompt-to-edit loop** drastically speeds up design iterations and also lowers the barrier for custom tailoring. Instead of manually tweaking dozens of measurements, a user could say "shorten the arms by 2 cm and widen the shoulders," and the AI will adjust the parametric model accordingly.

**Integration of Advanced Features:** The design platform will also handle the integration of functional subsystems like **active cooling units, sensor arrays, communication devices, or life support components**. We are approaching the suit as a **modular system**. In software, these will be represented as modules that can be toggled or configured. For instance, the active cooling module might consist of a network of microtubes and a battery-pack/pump unit; in the CAD model this could be an assembly added when the user selects "Cooling: ON". The AI assistant could activate these modules based on context (e.g. if the user says "configure for extreme heat"), or the user can manually add them. By planning these subsystems early, we ensure the digital model accounts for space, weight, and integration points for each feature. This is especially important for real-world prototyping - we'll design standardized attachment points (for example, ports where a cooling unit or air filter can plug in). The platform can simulate or at least validate some interactions (like checking there is room for a cooling vest under the suit, or that adding a tent attachment doesn't impede mobility).

**Game Engine and CAD Interoperability:** To provide an interactive user experience and rich visualization, our design software will leverage a **game engine (Unity or Unreal)** as the front-end, while linking to a CAD kernel in the back-end. The game engine gives us high-quality 3D rendering, VR/AR support, and a user-friendly GUI for manipulating the suit on a virtual avatar. Unity and Unreal also offer robust physics engines, which we might use to simulate how the suit behaves (e.g. draping of fabric, or mobility constraints) in a virtual environment. The **advantage of Unity/Unreal** is also deployment: we can package the application for multiple platforms (PC, Mac, potentially VR headsets) and easily distribute it via existing channels like Steam or the Microsoft Store, tapping into a broad user base of creators and early adopters. This means when we reach the stage of a public beta, **users worldwide can download our "Suit Studio" app from familiar stores**, lowering friction for community involvement. Under the hood, the parametric design computations might be handled by an embedded engine (like the FreeCAD kernel or a cloud service) - the game engine would send high-level design parameters or scripts to the CAD engine, get back an updated model mesh, and display it. Achieving smooth **CAD &lt;-&gt; game engine integration** is a technical challenge, but not unprecedented; we will investigate libraries or APIs for real-time CAD model updates in Unity. As a simpler interim step, we might use a library like OpenCASCADE for geometry and write custom code to update the Unity mesh. The end result is an app where a user can drag sliders or type commands to change the suit, see those changes immediately on a 3D avatar, and even experience it in AR/VR (for example, overlay the suit onto themselves via AR to check fit and look). This interactive approach will set our project apart and make the design process **highly user-centric and iterative**.

## Active Cooling, Heating, and Environmental Control

One of the hallmark features planned is **active thermal regulation** in the suits. Keeping the user comfortable and safe in extreme temperatures is critical for many target users (firefighters in high heat, astronauts in vacuum, divers in cold water, etc.). We are tracking the state-of-the-art closely to incorporate the best solutions:

- **Active Cooling:** As noted earlier, **personal cooling systems (PCS)** are already in use by military and industry, but we aim to refine and tailor them for our suits. Liquid cooling garments, which circulate chilled liquid through tubing, are a proven solution (NASA astronauts wear LCG vests under spacesuits). Oceanit's example shows that material science improvements (like more thermally conductive flexible tubing, and efficient micropumps) can dramatically boost cooling performance[\[36\]](https://oceanit.com/products/active-cooling-suit/#:~:text=1.%20A%20novel%2C%20thermally,between%20the%20body%20and%20tubing)[\[7\]](https://oceanit.com/products/active-cooling-suit/#:~:text=A%20key%20shortcoming%20of%20other,conductivity%20than%20PVC%20or%20Tygon). Our approach will consider such innovations - e.g. using high-conductivity silicone or graphene-infused tubes to extract heat from the body efficiently. We will also evaluate **phase change materials (PCM) vests** as a passive alternative or supplement - these vests contain packs that absorb heat by melting at a certain temperature, which can provide cooling without power for a limited time. A hybrid solution might work well: a PCM vest for short-term relief and a battery-powered liquid chiller for longer operations. Additionally, **air-cooled suits** (which blow cooled air through undergarments) exist and might be easier for certain contexts (like a suit plugged into an air hose in a vehicle or base camp). Our design will not commit to one method; rather, we'll ensure the suit can accommodate an **"active cooling module"** in different forms. For example, a firefighter version might have an attachment for a cooling vest insert or a quick-connect for a cooling hose during rehab breaks[\[37\]](https://shop.dqeready.com/public-safety/firefighter-rehab/core-cooling/#:~:text=Public%20Safety%20,with%20rest%20and%20hydration), whereas an astronaut version uses a closed-loop liquid system. By **designing a versatile cooling interface**, we can plug in new tech as it emerges (such as wearable thermoelectric coolers or advanced PCM packs).
- **Active Heating:** Conversely, for cold environments (deep water dives, high altitudes, winter disasters), active heating is needed. **Heated suits** and wetsuits are already on the market: these use thin, flexible heating elements (often carbon fiber or metal fiber pads) powered by rechargeable batteries[\[11\]](https://blog.wetsuitwearhouse.com/what-are-heated-wetsuits/#:~:text=Heated%20wetsuits%20are%20specially%20designed,comfortable%20temperature%20inside%20the%20suit)[\[38\]](https://blog.wetsuitwearhouse.com/what-are-heated-wetsuits/#:~:text=How%20Do%20Heated%20Wetsuits%20Work%3F). They often include thermostatic controls to adjust temperature. We will integrate heating elements into our suit design (especially the under-layer) for scenarios like Arctic expeditions or aiding people in blizzards. Ensuring even heat distribution and safety (no hot spots or fire risk) is key - techniques such as **printed flexible heaters** on textile or **self-regulating PTC heaters** can help. Our suit control system could intelligently manage heating and cooling, even switching between them as needed (for example, a climate refugee crossing a desert day and cold night could use the same suit to cool in daytime and heat at night).
- **Life Support & Filtration:** For specialized use (space, firefighting, polluted environments), the suit may need air filtration, oxygen supply, or hazardous substance protection. While full SCBA or space life support systems are beyond initial scope, we are reserving design space for these. For instance, the suit's helmet or hood could later house a **rebreather or filter unit**. Concept designs like the climate-adaptive clothing line included **smoke-filtering bandanas and integrated respirators**[\[39\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=Fireproof%20jackets.%20Smoke,we%20don%E2%80%99t%20stop%20climate%20change)[\[40\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=A%20jacket%20is%20fireproof%20and,removable%20side%20that%20filters%20water). We will design the suit's head/face enclosure to accept attachments like N95 filter inserts or small PAPR (powered air-purifying respirator) fans. Even for general consumers, having an **integrated mask for smoke or air pollution** could be a valuable feature given recent wildfire smoke events in many regions.
- **Sensors and Electronics:** In our roadmap, we also foresee embedding sensors (temperature, heart rate, GPS, air quality) and electronics for communications (radios, Bluetooth) into the suit. These can enhance safety (e.g., firefighter vitals monitor) and user experience (e.g., a diver's suit reporting depth and remaining oxygen via a HUD). The suit could act as a platform for **IoT in wearables**, though these aspects will be in later development phases.

In summary, our suit will evolve into a **smart wearable habitat** - capable of maintaining a stable microclimate for the wearer and providing protection from environmental hazards. By studying current SOTA products and research, we will implement these features in a modular, upgradable way. Importantly, we recognize that **starting simple is wise**: our initial prototypes might not include active cooling if that distracts from perfecting the base suit design. As the user noted, getting a solid suit model comes first, with active cooling integration as a next step. Our roadmap reflects this prioritization - focusing on core design and software before complex hardware integrations.

## Deployment, User Privacy, and Scalability

**Software Deployment:** As mentioned, our design platform will likely be delivered as a desktop application (with optional VR support) built on Unity or Unreal. This choice not only accelerates development (with rich UI and cross-platform support out-of-the-box) but also aligns with distribution avenues like **Steam or the Microsoft Store**, which we can leverage for reaching users. By using industry-standard engines, we get compatibility with high-end PCs as well as potentially cloud-streaming services (for example, someone could run the app on Nvidia GeForce Now cloud if their local machine is weak). We will maintain a **continuous integration pipeline** to push updates to users easily, treating the app much like a game in early access - frequently iterating based on feedback.

**Cloud vs Local Processing:** One challenge is that some features (like AI-driven design or 3D simulations) may require significant computation. We cannot assume every user has a powerful GPU or local AI accelerator. To ensure everyone can use the platform, we will implement a flexible compute model:  
\- By default, intensive tasks (e.g. generating a new suit via AI, running a physics sim, or processing a 3D body scan) can be offloaded to **cloud servers** that we operate. The client app will send the necessary data securely, the server does the heavy lifting (on GPU clusters or cloud services), and sends back results. This ensures even users on modest laptops get full functionality.  
\- However, we are mindful of privacy and offline use. If a user is dealing with sensitive data (say a **body scan which might be considered personal/medical data**), they may prefer to run the processing locally. We will therefore offer an **"offline mode" or local inference option** for key AI models. For instance, we might allow users to download a smaller version of our AI design model to run on their own GPU, or use local CPU rendering for CAD (albeit slower). Similarly, if a user doesn't want images of their body leaving their device (GDPR concerns), our app could perform the measurement extraction on-device. Embracing this **edge AI** approach has benefits for privacy and latency - processing data locally means sensitive biometrics never leave the user's control, helping comply with regulations like GDPR/HIPAA[\[41\]](https://symufolk.com/edge-ai-vs-cloud-ai/#:~:text=1,compliance%20with%20regulations%20like%20GDPR)[\[42\]](https://symufolk.com/edge-ai-vs-cloud-ai/#:~:text=,AI%20enhance%20data%20privacy). It also means the app can function in the field without internet (important if, say, a team in a disaster zone uses it on a local laptop to adjust suits).

We'll essentially adopt a **hybrid architecture**: real-time, privacy-critical, or offline scenarios use on-device computation, while complex or large-scale tasks leverage the cloud for efficiency[\[43\]](https://symufolk.com/edge-ai-vs-cloud-ai/#:~:text=When%20scaling%20AI%2C%20CTOs%20must,weigh%20the%20following%20factors)[\[44\]](https://symufolk.com/edge-ai-vs-cloud-ai/#:~:text=For%20many%20organizations%2C%20a%20hybrid,when%20the%20internet%20is%20available). This dual approach maximizes accessibility. From a technical standpoint, this means containerizing our AI models so they can run either in a cloud server or be downloaded and run with local libraries (possibly using frameworks like ONNX or TensorRT for local acceleration). Users who opt for cloud processing will have their data encrypted in transit and erased after processing on the server to maintain trust.

**Body Scanning and Custom Fit:** To achieve a perfect fit for every user, we plan to integrate **3D body scanning** capabilities. Many solutions now use just a smartphone camera to get a user's body measurements with centimeter accuracy[\[45\]](https://www.truetoform.fit/#:~:text=TrueToForm%20,to%20deliver%20the%20right%20fit)[\[46\]](https://7t.ai/client-successes/fit-freedom/#:~:text=Fit%20Freedom%20,measurements%20and%20recommending%20clothing%20sizes). Companies like 3DLook and Esenca offer AI-driven body measurement apps that can output dozens of measurements from a quick scan[\[47\]](https://3dlook.ai/#:~:text=3DLOOK%20uses%20AI%20for%203D,3D%20visualization%20of%20body%20progress)[\[48\]](https://www.prime-ai.com/en/media/primeai-ecommerce-sizing-solution-csf-c/#:~:text=Prime%20AI%20Size%20Finder%20vs,are%20intrusive%20and%20have). We will either partner with or replicate such technology within our platform. The idea is that a user (especially general consumer) could use their phone or webcam to scan themselves; the app then generates a personalized 3D avatar or a set of measurements, which feed into the suit model for an instant custom size. This is far more user-friendly than manual measuring and also ensures our suit pattern can be automatically adjusted to their body. Given privacy implications (body scans are sensitive data), we again will allow this to be done locally. Services like **Esenca emphasize GDPR-compliance and allow full data control** for clients[\[49\]](https://esencasizing.com/#:~:text=try%2C%20slashing%20return%20rates%20and,unlocking%20substantial%20cost%20savings), which will be our standard as well. If cloud is used (say to leverage a superior AI model), we will obtain user consent and employ strict anonymization (processing could even happen on the user's own cloud instance if they prefer). Technologically, the scanning might involve a neural network estimating a 3D mesh from photos - something we can either license or use open-source models for. The **output is a parameter set** (e.g. torso length, arm length, etc.) that plugs right into our parametric suit generator. This automation will make it possible for someone to literally "order" a custom-tailored suit through our system with minimal friction. In professional settings, it also ensures, for example, that each firefighter gets a suit fitted to their measurements rather than standard sizes, improving safety and comfort.

**Data Security and Compliance:** All user data - designs, body measurements, etc. - will be treated with utmost security. We'll follow best practices (encryption at rest and in transit, options to store data only locally, clear privacy policies). If any health-related data is involved (like precise body shape, which could be considered health data under some laws), we will comply with regulations like HIPAA in the US or analogous frameworks elsewhere. Users will have control: they can choose to share a design to the community or keep it private. Our ethos of openness applies to designs and research, not personal user data.

**Scalability:** As adoption grows, our cloud infrastructure will scale accordingly. Leveraging modern cloud platforms, we can auto-scale GPU instances during peak usage (perhaps using Kubernetes and cloud GPU pools). We'll monitor costs carefully - possibly implementing a fair-use or subscription model for heavy cloud use, while keeping basic usage free, to remain "lightly profitable" and sustainable. Given that Unity/Unreal allow easy updates, we can continuously improve the app with new features, eventually even pushing it to mobile AR platforms when feasible (imagine designing on an iPad with AR). By planning with scalability in mind (both in tech and business), we ensure the platform can serve everyone from a single indie designer to large organizations customizing suits for thousands of people.

## Feature Roadmap

We propose a **feature-based roadmap** (focusing on deliverables and capabilities rather than dates). This roadmap outlines the major milestones and functionalities we will develop as we progress:

- **Foundational Suit Model & Editor (MVP):** Develop the core **parametric 3D suit model** with basic variations (e.g. a base outdoor suit). Create a simple UI in the game engine to adjust key parameters (size measurements, a few options like adding a hood or not). Deliverable: _An interactive model viewer where users can input measurements and generate a custom-fitted basic suit._ This will validate our CAD integration and parametric approach.
- **AI Design Assistant Integration:** Incorporate the first version of the **natural language interface** for design. At this stage, the AI might handle a limited set of commands (e.g. "increase sleeve length" or "add a pocket on the left arm"). We will utilize an existing LLM (like GPT-4 or a fine-tuned variant) via our server to interpret these commands into model edits. Deliverable: _Users can modify the suit using text prompts in addition to GUI sliders._ This will demonstrate the feasibility of Codex/Gemini-style control over CAD and set the stage for more complex instructions.
- **Advanced Suit Variants & Modules:** Expand the suit designs to cover **multiple use-case variants**. This includes a **fire-resistant variant**, a **marine (dive) variant**, and a **space/high-altitude variant**. Each variant will introduce specialized components - e.g., the firefighter suit model with a thermal lining and oxygen tank mount, the dive suit with sealable zippers and heating elements, etc. We will also model the **attachment interfaces** for modules like cooling units, tent expansion, etc. These will largely be digital at this point (demonstrating in the CAD model how modules attach). Deliverable: _A library of suit templates for different scenarios, all editable within the platform._ Users can switch templates to see, for example, how a suit would look configured for firefighting versus for cold weather.
- **Material and Thermal Simulation:** Integrate basic **simulation capabilities** to validate design choices. This may include a thermal simulation (to estimate cooling/heating performance) and a stress test (to ensure seams and materials can handle loads). We might use simplified physics in Unity or hook into engineering solvers for this. While not full FEA, it will guide our physical prototyping. Additionally, allow the user to toggle different materials (from our sustainable material library) on the suit and see the impact (weight, protection level, cost). Deliverable: _Simulation feedback in the design app - e.g., an indicator of expected core temperature change with/without cooling system, or a warning if a certain design doesn't meet safety criteria._ This bridges the gap between pure design and functional performance.
- **Prototype Fabrication & Field Testing:** Using the designs, produce the **first physical prototypes** of the suit (likely one per variant). This is a real-world milestone rather than a software feature, but it's critical. We will fabricate suits using the chosen materials and integrate off-the-shelf components for cooling/heating as available. These prototypes will be tested with real users (firefighters, divers, etc., through partnerships or volunteer programs) for feedback on comfort, functionality, and any issues. Deliverable: _Physical Gen-1 prototype suits and a report on their performance._ The insights will loop back into improving the digital designs.
- **User Body Scanning & Fit Customization:** Integrate the **body scanning feature** into the app. By this point, we either have an in-app solution or an external app that feeds data to our system. Users can create a personal avatar from photos, then see the suit on that avatar for fit visualization. We also implement **automatic pattern generation** - i.e., from the adjusted 3D model, the system can output 2D patterns or CNC instructions for fabric cutting. Deliverable: _End-to-end custom fitting workflow:_ the user scans themselves, the suit model auto-adjusts, and the app can output files to manufacture that custom suit. This is key for later scaling manufacturing or letting makers download patterns to sew their own (in line with open-source ethos).
- **Full Module Functionality (Cooling, Tent, etc.):** Develop and integrate the **active modules** into both the digital platform and physical prototypes. This includes the _Active Cooling Unit_ (a working liquid cooling system or PCM packs that can be attached to the suit), the _Tent Module_ (a lightweight canopy or sleeping-bag attachment that can zip onto the suit, akin to the jacket-tent concept[\[50\]](https://www.fastcompany.com/3055886/this-jacket-designed-for-refugees-transforms-into-a-tent#:~:text=During%20the%20day%2C%20this%20is,protects%20someone%20from%20the%20elements)), and any _power systems_ needed (battery packs, solar charging on the suit if possible). We will simulate these in software (e.g., show how the tent deploys from the suit model) and build real add-ons to test. Deliverable: _Multi-functional suit demonstration:_ a user can go from wearing the suit to setting it up as a shelter, or engage a cooling system on a firefighter suit in a heat chamber test, etc. This stage proves our suit's versatility and moves it closer to the "hurricane tent replacement" ideal.
- **Polished Product Release (v1.0):** Incorporate all feedback, refine the software UI/UX, ensure stability, and prepare a polished release of the platform. At this stage, the platform will include: the AI design assistant with robust understanding of suit-specific commands, a range of suit templates, full custom sizing, and links to either our production partners or DIY instructions for getting the suit made. We will also release extensive documentation (a "world-class" README and user manual, of which this document is the genesis) and our **open-source repositories** for core components. Deliverable: _Version 1.0 of the Suit Design Studio software_ (available for download), and initial availability of physical suits or kits for early adopters. This marks the transition from R&D to a deployable product.
- **Community and Humanitarian Deployment:** Post-launch, we will focus on building a **community** around the project - encouraging researchers, makers, and early users to contribute improvements or new ideas (for example, a user might design a new module like a **radiation sensor**, which can be shared with all). We'll also start executing our **humanitarian distribution strategy**: partnering with NGOs to trial the suits in disaster relief operations, donating suits to firefighter units in under-resourced areas, etc. This is less a technical milestone and more of scaling our impact. It will involve setting up production for larger batches at extremely low cost, possibly through sponsors or grants. Deliverable: _Successful pilot programs deploying suits for humanitarian aid_, plus an active developer community extending the platform (with perhaps quarterly community showcases or hackathons to keep innovation flowing).

Throughout this roadmap, each step builds on the previous, adding layers of capability - from a basic digital model to a fully realized product ecosystem. We have intentionally structured it so that we deliver value at each stage (e.g., even the MVP could be useful for custom clothing makers, and intermediate prototypes benefit our partners). Importantly, the **features are prioritized by necessity and feasibility**: core design first, then AI and variant expansions, and hardware modules later once the design is solid. This ensures we tackle risks in order (e.g., verifying we can actually make a comfortable suit before adding complex cooling tech to it).

## Key Deliverables and Technical Strategy

To successfully realize the above vision, we have identified a set of **key deliverables** along with the technologies and methods we will use to achieve them:

- **Parametric Suit CAD Model Library:** _Deliverable:_ A collection of digital suit models (in CAD format) that cover the base design and all variant components (helmet, vest, tent extension, etc.). _Technology:_ We will use a CAD modeling kernel (likely **OpenCASCADE or similar** via Python scripts, or a parametric modeler like **FreeCAD**). The models will be defined with a clear set of parameters for sizing and feature toggles. We will also maintain these models under version control (possibly using Git for CAD files with a plugin) since they are as central to the project as code. The parametric definitions might be exposed in a JSON or script form so the AI can manipulate them easily.
- **Suit Design Studio Application:** _Deliverable:_ The interactive application built on Unity/Unreal for editing and visualizing suits. _Technology:_ **Unity 3D** (C#) or **Unreal Engine** (C++/Blueprints) will be used to create the UI, 3D viewport, and logic for user interactions. We'll integrate with the CAD backend via a plugin or by running a background service that the app communicates with (for instance, the app sends param updates to a small server that generates the new mesh with OpenCASCADE and returns it). For AI integration, the app will connect to our AI service or local model (could be via a simple REST API or an SDK if using something like OpenAI's API). We'll also make use of Unity's asset system to manage 3D assets of modules (e.g. we might model the cooling pump as a separate object that Unity can attach when needed). By using a game engine, we also get **cross-platform deployment** easily, and can incorporate nice-to-haves like animation (e.g. animating how the tent deploys, or how the suit behaves on a moving human model).
- **AI Design Back-end:** _Deliverable:_ The AI agent that interprets text commands and modifies the suit model. _Technology:_ Initially, this could be built on **OpenAI GPT-4/Codex** or similar via their API, using prompts crafted to translate natural language into our parametric instructions. We will define a "CAD command language" for our suits (for example, a JSON schema or a small DSL where you can say {action: "extrudePocket", position: "left_arm", size: \[10,5,2\]} or similar). The LLM will output commands in that format, which our app then executes on the model. Over time, we might fine-tune our own model (possibly using **Llama 2** or waiting for **Google Gemini** which promises powerful multi-modal capabilities) to reduce dependency on external APIs and allow offline usage. We will gather training data from our design iterations (every time a human designer makes a change, we can log a description and the resulting param change to improve the AI). Our approach will be incremental: start with simple prompt engineering and gradually develop a specialized **CAD-LLM model** - Autodesk's research on CAD LLMs[\[51\]](https://www.research.autodesk.com/publications/ai-lab-cad-llm/#:~:text=CAD,them%20to%20manipulate%20engineering%20sketches) and projects like Text2CAD will guide us.
- **Body Scanning Module:** _Deliverable:_ A component of the platform (or a companion mobile app) that captures user measurements. _Technology:_ If partnering with an existing solution (like 3DLook or Esenca), the deliverable could be an **SDK integration** in our app that takes camera input and returns measurements. If building in-house, we'll use **computer vision and deep learning** - likely a pre-trained model that can predict a 3D mesh from two photos (front and side) or a short AR scan. There are open research models for this (e.g., using **SMPL body model** fitting). We'll wrap it into a user-friendly flow. Ensuring this is robust across body types and phone cameras is important, so we'll test with diverse users. Output will be automatically fed into the parametric model to adjust it.
- **Physical Prototypes (Suits and Modules):** _Deliverable:_ Real, wearable suits incorporating our design features, plus attachable modules like cooling vests, etc. _Technology & Approach:_ We will likely use a combination of **traditional sewing for garment parts** and **rapid prototyping for hardware**. For instance, we might 3D-print connectors or housings for the cooling system, use laser-cut fabrics for precision, and incorporate off-the-shelf electronics (like Arduino-based controllers for heating elements). We'll leverage modern **fabrication labs (FabLab)** resources: CNC cutters for patterns, industrial sewing machines, and possibly novel fabrication like printed electronics for sensor integration. Materials will be sourced according to our sustainable specs (recycled or organic textiles, etc.), and we'll document the bill of materials. Each prototype generation will be tested in conditions simulating use (thermal labs for heat, pressure tests for dive gear, etc.). We'll use those results to refine the design (closing the CAD ↔ physical loop).
- **Cloud Infrastructure & API:** _Deliverable:_ A cloud-based system to handle heavy computations and AI processing for users. _Technology:_ We will use a reliable cloud provider (AWS, Azure, or Google Cloud) to host our services. This includes an **AI inference server** (running on GPU instances) for the design assistant and possibly the body scan processing, and a **CAD generation server** if needed (though ideally CAD can mostly be local, heavy tasks like generative design might go here). We will design RESTful APIs or WebSocket services for the app to communicate with these. To handle scaling, we'll use containerization (Docker) and orchestrators (Kubernetes) - allowing dynamic scaling of the AI inference pods based on load. We'll implement security with authentication and perhaps end-to-end encryption for any sensitive data that must go to cloud. Over time, we may also explore **edge computing** solutions - for example, distributing some computations to edge servers or allowing volunteers to contribute computing (akin to Folding@home but for suit simulations), though that's an exploratory idea.
- **Documentation and Open-Source Repositories:** _Deliverable:_ Comprehensive documentation (user guides, developer docs) and open-source code release for components of the project. _Approach:_ We will maintain a **GitHub (or similar) repository** for the core code (except perhaps proprietary bits if any, but ideally all open). Documentation will include this README (as a living document updated with progress), a technical whitepaper (covering our research findings, e.g., comparison of cooling methods, material tests), and usage examples. We'll encourage contributions by providing clear contribution guidelines and a roadmap in the repo so others can align their PRs with our plan. Key design files (like CAD models) might be hosted in a data repository or as part of the main repo if manageable. By deliverable of v1.0, the project should be in a state where someone else could fork it and build upon it - that ensures transparency and aligns with our ethos.

In terms of **target technologies**, here's a summary mapping: CAD/B-rep modeling (FreeCAD/OpenCASCADE + Python), Game engine (Unity/Unreal with C# or C++), AI (initially GPT-4 API, moving to open models on PyTorch/TensorFlow, possibly running with ONNX for local), CV (OpenCV + neural net models for body scan), Cloud (Docker/K8s on AWS/Azure with GPU instances for AI, maybe using services like Nvidia GPU Cloud or Lambda Labs for flexibility). We will keep an eye on emerging tech; for example, if a new open-source LLM like **GPT-4 X** or **Gemini** becomes available that can handle text+images+context better, we'll integrate it to improve the AI assistant's capability to understand design context or even user sketches. Similarly, advancements in AR toolkits might let us project the suit onto the user's mirror view in real-time, enhancing fitting sessions.

By delivering on these components step by step, we aim to create a **world-class platform for adaptive suit design** that not only pushes the state-of-the-art in several domains (wearable tech, AI-assisted CAD, sustainable materials) but also delivers tangible products that save lives and help people adapt to our changing world. Each deliverable has been scoped with technologies that are either currently available or on the near horizon of maturity, giving us confidence that this ambitious roadmap is achievable. Together, this will culminate in an innovative, ethically-grounded solution: **wearable shelters and smart protective suits for all who need them**, designed and delivered using the best tools modern science and technology can offer.

**Sources:** _(The following sources provide context and support for the research and technology assumptions in this document.)_

[\[6\]](https://oceanit.com/products/active-cooling-suit/#:~:text=Current%20PCSs%20fall%20into%20three,cooled%2C%20and%20phase%20change%20vests)[\[7\]](https://oceanit.com/products/active-cooling-suit/#:~:text=A%20key%20shortcoming%20of%20other,conductivity%20than%20PVC%20or%20Tygon) Oceanit - _Advanced Cooling Vest innovation demonstrating multi-method personal cooling and novel high-conductivity materials._  
[\[5\]](https://oceanit.com/products/active-cooling-suit/#:~:text=Current%20PCSs%20fall%20into%20three,cooled%2C%20and%20phase%20change%20vests) Oceanit - _Classification of Personal Cooling Systems (liquid, air, phase-change) in current use._  
[\[20\]](https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear#:~:text=There%20are%20four%20essential%20elements,sustainable%20protective%20clothing%20supply%20chain)[\[24\]](https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear#:~:text=Well,evidence%20of%20the%20product%27s%20origins) TenCate Fabrics - _Sustainable PPE fabrics, including recycled polyester and emerging recycled aramid fibers._  
[\[12\]](https://sheltersuit.com/en-us/#:~:text=The%20Sheltersuit%20is%20a%20wind,bag%2C%20and%20a%20duffel%20bag)[\[13\]](https://sheltersuit.com/en-us/#:~:text=The%20products%20we%20provide%20are,upcycled%20materials%20to%20reduce%20waste) Sheltersuit Foundation - _Example of a humanitarian multi-function suit (jacket + sleeping bag) made from upcycled materials._  
[\[3\]](https://medium.com/the-journal-of-critical-space-studies/pacific-spaceflight-ab25c5b4347c#:~:text=Unlike%20Dolgov%20and%20other%20pressure,NASA%20or%20private%20aerospace%20companies)[\[4\]](https://medium.com/the-journal-of-critical-space-studies/pacific-spaceflight-ab25c5b4347c#:~:text=falling%20out) _Medium (C. Smith DIY spacesuit)_ - Illustrative case of dramatically reducing spacesuit cost and the philosophy of democratizing space technology.  
[\[30\]](https://arxiv.org/html/2409.17106v1#:~:text=Prototyping%20complex%20computer,draw%20two%20circles%20with)[\[31\]](https://arxiv.org/html/2409.17106v1#:~:text=Currently%2C%20there%20are%20works%20on,dagger%20%2041%20https%3A%2F%2Fgithub.com%2FKittyCAD%2Fmodeling) Text2CAD (2024) - _Research introducing the first text-to-parametric-CAD framework, validating our AI design approach._  
[\[33\]](https://zoo.dev/text-to-cad#:~:text=Turning%20thoughts%20into%20complex%20mechanical,designs)[\[34\]](https://zoo.dev/text-to-cad#:~:text=After) Zoo Design Studio - _Commercial example of generating editable CAD models from simple text prompts ("Text-to-CAD" with prompt-to-edit functionality)._  
[\[41\]](https://symufolk.com/edge-ai-vs-cloud-ai/#:~:text=1,compliance%20with%20regulations%20like%20GDPR)[\[42\]](https://symufolk.com/edge-ai-vs-cloud-ai/#:~:text=,AI%20enhance%20data%20privacy) Symufolk (Edge vs Cloud AI) - _Discussion on how on-device AI improves privacy and GDPR compliance by keeping data local._  
[\[48\]](https://www.prime-ai.com/en/media/primeai-ecommerce-sizing-solution-csf-c/#:~:text=Prime%20AI%20Size%20Finder%20vs,are%20intrusive%20and%20have) 3DLook/Esenca - _Overview of AI body scanning for custom-fit apparel, used in fashion and protective gear to get precise measurements._  
[\[10\]](https://blog.wetsuitwearhouse.com/what-are-heated-wetsuits/#:~:text=Quadri%20Abdur%20%C2%A0%C2%A0%20February%2018th%2C,2025%C2%A0%C2%A0%20Posted%20In%3A%20%20161)[\[11\]](https://blog.wetsuitwearhouse.com/what-are-heated-wetsuits/#:~:text=Heated%20wetsuits%20are%20specially%20designed,comfortable%20temperature%20inside%20the%20suit) Wetsuit Wearhouse - _Description of battery-powered heated wetsuits, an analog for our active heating integration._  
[\[1\]](https://comex.fr/en/news-en/new-intelligent-materials-for-future-space-suits/#:~:text=The%20objective%20of%20the%20project,and%20the%20Austrians%20from%20OeWF)[\[2\]](https://comex.fr/en/news-en/new-intelligent-materials-for-future-space-suits/#:~:text=identify%20materials%20capable%20of%20resisting,the%20many%20external%20aggressions) COMEX/ESA PEXTEX - _Space industry project identifying new textiles for lunar/Martian suits, highlighting extreme environment requirements._  
[\[14\]](https://www.fastcompany.com/3055886/this-jacket-designed-for-refugees-transforms-into-a-tent#:~:text=The%20jacket%20is%20made%20from,like%20fabric%20used%20in%20envelopes)[\[15\]](https://www.fastcompany.com/3055886/this-jacket-designed-for-refugees-transforms-into-a-tent#:~:text=Image) FastCompany (Refugee Jacket-Tent) - _Innovative design using Tyvek for a jacket that becomes a tent, emphasizing low-cost, recyclability, and multi-use design._  
[\[16\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=A%20jacket%20is%20fireproof%20and,removable%20side%20that%20filters%20water)[\[19\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=But%20after%20the%20creative%20process,%E2%80%9D) FastCompany (Climate Change Apparel) - _Concept line "Unfortunately, Ready to Wear" with fireproof, protective clothing items, noting their current relevance due to climate crises._

[\[1\]](https://comex.fr/en/news-en/new-intelligent-materials-for-future-space-suits/#:~:text=The%20objective%20of%20the%20project,and%20the%20Austrians%20from%20OeWF) [\[2\]](https://comex.fr/en/news-en/new-intelligent-materials-for-future-space-suits/#:~:text=identify%20materials%20capable%20of%20resisting,the%20many%20external%20aggressions) PEXTEX Project - COMEX

<https://comex.fr/en/news-en/new-intelligent-materials-for-future-space-suits/>

[\[3\]](https://medium.com/the-journal-of-critical-space-studies/pacific-spaceflight-ab25c5b4347c#:~:text=Unlike%20Dolgov%20and%20other%20pressure,NASA%20or%20private%20aerospace%20companies) [\[4\]](https://medium.com/the-journal-of-critical-space-studies/pacific-spaceflight-ab25c5b4347c#:~:text=falling%20out) The World's Only Homemade Spacesuit Is About To Get Its First Life-or-Death Test | by Laika Valentina | The Journal of Critical Space Studies | Medium

<https://medium.com/the-journal-of-critical-space-studies/pacific-spaceflight-ab25c5b4347c>

[\[5\]](https://oceanit.com/products/active-cooling-suit/#:~:text=Current%20PCSs%20fall%20into%20three,cooled%2C%20and%20phase%20change%20vests) [\[6\]](https://oceanit.com/products/active-cooling-suit/#:~:text=Current%20PCSs%20fall%20into%20three,cooled%2C%20and%20phase%20change%20vests) [\[7\]](https://oceanit.com/products/active-cooling-suit/#:~:text=A%20key%20shortcoming%20of%20other,conductivity%20than%20PVC%20or%20Tygon) [\[8\]](https://oceanit.com/products/active-cooling-suit/#:~:text=Oceanit%20is%20developing%20advanced%20Liquid,operating%20in%20harsh%20thermal%20environments) [\[9\]](https://oceanit.com/products/active-cooling-suit/#:~:text=Oceanit%E2%80%99s%20%E2%80%9CSuper%20Cool%20Vest%E2%80%9D%20was,by%20using%20four%20key%20innovations) [\[36\]](https://oceanit.com/products/active-cooling-suit/#:~:text=1.%20A%20novel%2C%20thermally,between%20the%20body%20and%20tubing) Active Cooling Suit - Oceanit

<https://oceanit.com/products/active-cooling-suit/>

[\[10\]](https://blog.wetsuitwearhouse.com/what-are-heated-wetsuits/#:~:text=Quadri%20Abdur%20%C2%A0%C2%A0%20February%2018th%2C,2025%C2%A0%C2%A0%20Posted%20In%3A%20%20161) [\[11\]](https://blog.wetsuitwearhouse.com/what-are-heated-wetsuits/#:~:text=Heated%20wetsuits%20are%20specially%20designed,comfortable%20temperature%20inside%20the%20suit) [\[38\]](https://blog.wetsuitwearhouse.com/what-are-heated-wetsuits/#:~:text=How%20Do%20Heated%20Wetsuits%20Work%3F) What Are Heated Wetsuits & How Do They Work? - Wetsuit Wearhouse Blog

<https://blog.wetsuitwearhouse.com/what-are-heated-wetsuits/>

[\[12\]](https://sheltersuit.com/en-us/#:~:text=The%20Sheltersuit%20is%20a%20wind,bag%2C%20and%20a%20duffel%20bag) [\[13\]](https://sheltersuit.com/en-us/#:~:text=The%20products%20we%20provide%20are,upcycled%20materials%20to%20reduce%20waste) [\[26\]](https://sheltersuit.com/en-us/#:~:text=The%20outer%20shell%20is%20made,and%20contains%20an%20integrated%20scarf) [\[27\]](https://sheltersuit.com/en-us/#:~:text=labor%20force%20and%20made%20out,upcycled%20materials%20to%20reduce%20waste) Sheltersuit Foundation

<https://sheltersuit.com/en-us/>

[\[14\]](https://www.fastcompany.com/3055886/this-jacket-designed-for-refugees-transforms-into-a-tent#:~:text=The%20jacket%20is%20made%20from,like%20fabric%20used%20in%20envelopes) [\[15\]](https://www.fastcompany.com/3055886/this-jacket-designed-for-refugees-transforms-into-a-tent#:~:text=Image) [\[50\]](https://www.fastcompany.com/3055886/this-jacket-designed-for-refugees-transforms-into-a-tent#:~:text=During%20the%20day%2C%20this%20is,protects%20someone%20from%20the%20elements) This Jacket Designed For Refugees Transforms Into A Tent - Fast Company

<https://www.fastcompany.com/3055886/this-jacket-designed-for-refugees-transforms-into-a-tent>

[\[16\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=A%20jacket%20is%20fireproof%20and,removable%20side%20that%20filters%20water) [\[17\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=But%20after%20the%20creative%20process,%E2%80%9D) [\[18\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=This%20new%20clothing%20line%20is,life%20after%20apocalyptic%20climate%20change) [\[19\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=But%20after%20the%20creative%20process,%E2%80%9D) [\[39\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=Fireproof%20jackets.%20Smoke,we%20don%E2%80%99t%20stop%20climate%20change) [\[40\]](https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change#:~:text=A%20jacket%20is%20fireproof%20and,removable%20side%20that%20filters%20water) This new clothing line is designed for life after apocalyptic climate change - Fast Company

<https://www.fastcompany.com/90303543/this-new-clothing-line-is-designed-for-life-after-apocalyptic-climate-change>

[\[20\]](https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear#:~:text=There%20are%20four%20essential%20elements,sustainable%20protective%20clothing%20supply%20chain) [\[21\]](https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear#:~:text=REPREVE%C2%AE%20recycled%20polyester%20fibres%20in,remaining%20durable%20during%20industrial%20laundering) [\[22\]](https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear#:~:text=,compares%20favourably%20against%20virgin%20polyester) [\[23\]](https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear#:~:text=recycled%20plastic%20bottles%2C%20REPREVE%C2%AE%20fibres,remaining%20durable%20during%20industrial%20laundering) [\[24\]](https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear#:~:text=Well,evidence%20of%20the%20product%27s%20origins) [\[25\]](https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear#:~:text=Recycled%20aramids%20are%20already%20regenerated,in%20the%20protective%20clothing%20industry) Integrating sustainable fabrics in PPE workwear clothing

<https://eu.tencatefabrics.com/sustainable-fabrics-in-ppe-workwear>

[\[28\]](https://arxiv.org/html/2409.17106v1#:~:text=Computer,iteratively%20refine%20the%20final%20models) [\[29\]](https://arxiv.org/html/2409.17106v1#:~:text=Despite%20their%20capabilities%2C%20modern%20CAD,parametric%20CAD%20generation%2C%20making%20it) [\[30\]](https://arxiv.org/html/2409.17106v1#:~:text=Prototyping%20complex%20computer,draw%20two%20circles%20with) [\[31\]](https://arxiv.org/html/2409.17106v1#:~:text=Currently%2C%20there%20are%20works%20on,dagger%20%2041%20https%3A%2F%2Fgithub.com%2FKittyCAD%2Fmodeling) [\[32\]](https://arxiv.org/html/2409.17106v1#:~:text=Currently%2C%20there%20are%20works%20on,body%2C%20and) Text2CAD: Generating Sequential CAD Models from Beginner-to-Expert Level Text Prompts

<https://arxiv.org/html/2409.17106v1>

[\[33\]](https://zoo.dev/text-to-cad#:~:text=Turning%20thoughts%20into%20complex%20mechanical,designs) [\[34\]](https://zoo.dev/text-to-cad#:~:text=After) ML CAD Model Generator | Create CAD Files With Text | Zoo

<https://zoo.dev/text-to-cad>

[\[35\]](https://medium.com/@gwrx2005/ai-driven-design-tool-for-blueprints-and-3d-models-a9145f5ee537#:~:text=AI,3D%20model%20or%20drawing) AI-Driven Design Tool for Blueprints and 3D Models - Medium

<https://medium.com/@gwrx2005/ai-driven-design-tool-for-blueprints-and-3d-models-a9145f5ee537>

[\[37\]](https://shop.dqeready.com/public-safety/firefighter-rehab/core-cooling/#:~:text=Public%20Safety%20,with%20rest%20and%20hydration) Public Safety - Firefighter Rehab - Core Cooling Equipment - DQE

<https://shop.dqeready.com/public-safety/firefighter-rehab/core-cooling/>

[\[41\]](https://symufolk.com/edge-ai-vs-cloud-ai/#:~:text=1,compliance%20with%20regulations%20like%20GDPR) [\[42\]](https://symufolk.com/edge-ai-vs-cloud-ai/#:~:text=,AI%20enhance%20data%20privacy) [\[43\]](https://symufolk.com/edge-ai-vs-cloud-ai/#:~:text=When%20scaling%20AI%2C%20CTOs%20must,weigh%20the%20following%20factors) [\[44\]](https://symufolk.com/edge-ai-vs-cloud-ai/#:~:text=For%20many%20organizations%2C%20a%20hybrid,when%20the%20internet%20is%20available) Edge AI Vs Cloud AI: What CTOs Need To Know Before Scaling

<https://symufolk.com/edge-ai-vs-cloud-ai/>

[\[45\]](https://www.truetoform.fit/#:~:text=TrueToForm%20,to%20deliver%20the%20right%20fit) TrueToForm - 3D Body Scan for Measurements

<https://www.truetoform.fit/>

[\[46\]](https://7t.ai/client-successes/fit-freedom/#:~:text=Fit%20Freedom%20,measurements%20and%20recommending%20clothing%20sizes) Fit Freedom - 7T, Inc. | Dallas

<https://7t.ai/client-successes/fit-freedom/>

[\[47\]](https://3dlook.ai/#:~:text=3DLOOK%20uses%20AI%20for%203D,3D%20visualization%20of%20body%20progress) 3DLOOK - AI-powered 3D body scanning solution

<https://3dlook.ai/>

[\[48\]](https://www.prime-ai.com/en/media/primeai-ecommerce-sizing-solution-csf-c/#:~:text=Prime%20AI%20Size%20Finder%20vs,are%20intrusive%20and%20have) Prime AI Size Finder vs. Body Scanning Tools for Accurate Clothing ...

<https://www.prime-ai.com/en/media/primeai-ecommerce-sizing-solution-csf-c/>

[\[49\]](https://esencasizing.com/#:~:text=try%2C%20slashing%20return%20rates%20and,unlocking%20substantial%20cost%20savings) Esenca | The 3D Body Measurement Solution for Better Sizing

<https://esencasizing.com/>

[\[51\]](https://www.research.autodesk.com/publications/ai-lab-cad-llm/#:~:text=CAD,them%20to%20manipulate%20engineering%20sketches) CAD-LLM: Large Language Model for CAD Generation

<https://www.research.autodesk.com/publications/ai-lab-cad-llm/>




Further references:

Layer
	
Component
	
Material/Configuration
	
Mechanical Properties
	
Joint/Limb Association
	
Output Files
	
Integration Methodology
	
Validation Metrics
	
Source
Undersuit
	
Liquid loops / Tubes
	
Liquid (water/glycol); polymer tubes; high-conductivity silicone or graphene-infused tubes; NASA LCVG archetype
	
Cooling capacity over time; thermal conductivity (~50% better in polymers); 30% outperformance in heat extraction; flow: 1.5 L/min
	
Armpits, neck, groin, major limbs; trunk-to-limb loops; body-wide thermal zones
	
Suit Circuit schema (JSON); liquid-loop manifest; 2D patterns (DXF, SVG, PDF)
	
Modular routing; graph-based tubing solver (segments=edges, fittings=nodes); Unity/Unreal brush interface
	
Skin temp deltas; pump flow requirements; thermal flow feedback; load balancing (zone_flow_rate)
	
[1-5]
Undersuit
	
PCM packs / Phase-change inserts
	
Passive cooling; microcapsules in textile; latent-capacity packs
	
Thermal resistance; heat absorption via melting; passive conduction; flow rates reported as zero
	
Regional priority (armpits, neck, groin); thermal zones proportional to heat load
	
Cooling manifest (JSON); 2D patterns (DXF, SVG, PDF)
	
Hybrid solution (PCM + liquid chiller); programmable API; global capacity overrides
	
Pressure comfort levels; core temperature indicators; zone capacity matching
	
[3, 5]
Hard Shell
	
Rigid armor plates / Segmented panels
	
Single-polymer; bio-fiber (basalt/aramid); bio-based recycled aramids; recyclable polymers; watertight triangle mesh
	
Constant-thickness offset shelling; impact stress resistance; hinge allowance; thickness profile (min/max/mean)
	
Shoulder, elbow, knee, hip, spine; anatomical modules (spine, scapula, humerus)
	
STL, STEP, GLB, FBX; shell_layer.npz; manifest.json; Parametric CAD models
	
Parametric armor rigs; Blender Geometry Nodes; Text2CAD/AI Python scripts; Unity/Unreal integration
	
Clearance map (0°/45°/90° flexion); watertightness; bone-naming preservation; joint cone overlays
	
[1, 2, 4, 6-8]
Cloth
	
Spandex bodysuit / Spun yarn
	
Lycra; negative ease (~10% reduction); elastic, insulative, pressure-mapped zones
	
Stretch ratio (current/rest length); shear; anisotropic elastic shell; fabric anisotropy vector
	
Center spine, chest, shoulders, quads, hamstrings, knee cap, elbow crease
	
SVG/DXF patterns; strain heatmap (strain.png); PDF tiling (A4/A0)
	
FEM/Cloth solver (Blender/Houdini); SGD optimization; SOTA ST-Tech Cubic Barrier Solver
	
Max seam-adjacent strain; subjective comfort scores; pressure gradients; curvature-guided seams
	
[1-3, 9, 10]
Cloth
	
Soft-layer / Canopy
	
Tyvek (waterproof); upcycled tent fabric; recycled polyester; TENCEL™ Lyocell
	
Stretch/shear ratios (inferred from draping); seam allowances; fold paths
	
Shoulder (packed behind shoulders); landmarks: c7_vertebra, sternum, acromion
	
suit_with_tent.glb; printable PDF fold sequence; bundle.json
	
Deployment planning via build_deployment_kinematics(); Unity/Unreal physics
	
Anchor placement; fold-path validation; anchor coverage; fold connectivity
	
[4, 11]
[1] ROADMAP(1).md
[2] les - seameinit.pdf
[3] smii - seameinit.pdf
[4] Next-Generation Adaptive Suit Platform – Vision and Roadmap.pdf
[5] cooling.md
[6] engine_integration.md
[7] hard_layer_shell.md
[8] hard_shell_clearance.md
[9] les - Glove seam optimisation papers.pdf
[10] smii - Auto-rigging cloth simulation.pdf
[11] tent.md
