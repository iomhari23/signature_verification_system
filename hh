For a live project with 50,000 users and high-concurrency "shift change" peaks, the best approach must pivot from "accuracy only" to "Efficiency + Scalability."
Here is the re-checked, professional-grade architecture for your hospital system:
1. The Data Architecture (Hybrid Storage)
At 50k users, you cannot rely solely on a single JSON blob for everything.
The "Master" Record (MySQL): Store a single Centroid Vector (the average of your 20 baseline signatures) in a fixed-length BINARY or BLOB column.
The "Variation Gallery" (MySQL JSON): Keep the 20 individual vectors in a JSON column.
Why? In 90% of cases, the user will match the Master vector. Comparing 1-to-1 is much faster than 1-to-20. You only open the "Gallery" JSON if the Master check is "borderline."
2. The Verification Logic (The "Tiered" Approach)
To maintain <500ms speed during 50k-user peaks, use a tiered check:
Tier 1: Fast Match. Compare Canvas Vector vs. Master Centroid. If Similarity > 0.95, Approve Instantly.
Tier 2: Variation Match. If Similarity is 0.80–0.95, pull the Gallery JSON. Compare against the 20 variations. If it matches one, Approve.
Tier 3: Active Learning Update. If the match is Tier 1 (High Confidence), trigger an asynchronous background task to update the Gallery with the new variation.
3. Computation Strategy (The "Edge" Shift)
With 50k users, your server will crash if it does all the image processing.
Client-Side (Canvas): Use JavaScript to perform Skeletonization and Normalization (Resizing to 224x224) directly in the browser/app.
Server-Side (Inference): Convert your .pth model to ONNX format. ONNX is significantly faster for "Inference only" (running the model) than PyTorch and handles multiple requests better.
4. Database Optimization for 50k Users
Indexing: Ensure client_srno is the Clustered Index.
Caching: Use Redis to store the Master Vectors of the "Expected Shift." (e.g., if 500 doctors are scheduled for the 8 AM shift, preload their vectors into Redis at 7:55 AM). This makes "Check-in" feel like magic.
5. Managing the "Active Learning" Risk
With 50k users, you face the "Template Poisoning" risk—where a user accidentally lets someone else sign, and the system "learns" the wrong signature.
The 3-Tier Threshold:
Accept & Learn: Similarity > 0.96 (Update Gallery).
Accept Only: Similarity 0.85 - 0.95 (Do NOT update Gallery).
Reject: Similarity < 0.85.
This ensures your baseline remains "Pure" while still being flexible.
Final "Best Approach" Summary
Baseline: 20 signatures during onboarding.
Storage: 1 Master Vector (Binary) + 20 Gallery Vectors (JSON) in MySQL.
Engine: Siamese Network exported to ONNX.
Flow: Client-side processing -> Master Match -> (Optional) Gallery Match -> Background Update.
How are you planning to handle the "Onboarding"? Will you have a dedicated kiosk for the 15-20 signature collection?





Master vector definated this
In your system, the Master Vector is the mathematical "Gold Standard" for a specific user. It is the single most important piece of data for speed and accuracy at a 50k-user scale.
1. What it is (The "Centroid")
Instead of being a single signature, the Master Vector is the average of your 15–20 baseline signatures.
The Math: If your vector has 128 dimensions, you take the average of the 1st number across all 20 signatures, then the 2nd number, and so on.
The Result: You get a single 128-dimensional vector that sits exactly in the "middle" of that person's signature style.
2. Why it is the "Best Approach" for 50k Users
Extreme Speed: Comparing 1 vector (Live) to 1 vector (Master) is roughly 20x faster than checking the entire Gallery. In a shift change with hundreds of people signing at once, this saves your server from lagging.
Noise Reduction: A single "shaky" signature in the baseline is balanced out by the other 19. The Master Vector represents the "cleanest" version of the user's intent.
Initial Filter: 90% of the time, the Master Vector will be enough to verify the user. You only "unlock" the heavy JSON Gallery if the Master Vector check is uncertain.
3. How to define it in your Workflow
Creation: During onboarding, you collect 20 signatures. You run them through your .pth model to get 20 vectors. You average them to create the Master Vector.
Storage: Store this Master Vector in a dedicated column in MySQL (type BINARY(512) or BLOB). Do not hide it inside a JSON array; it needs to be "ready to use."
Updating: Unlike the Gallery (which updates frequently), the Master Vector should only be updated rarely—perhaps once a month or if the staff member’s signature style changes significantly.
4. Verification Logic with the Master
Step 1: Compare Live_Vector vs Master_Vector.
Step 2: If Similarity > 0.95, they are in. (Stop here).
Step 3: If Similarity is 0.80 - 0.94, only then fetch the JSON Gallery to see if this "messy" signature matches a previous "messy" variation they’ve done before.
Would you like the specific Python code (using NumPy) to calculate this "Average" Master Vector from your 20 samples?





accuracy
For a high-traffic system with 50,000 users, accuracy is measured by the balance between False Acceptance Rate (FAR) (security) and False Rejection Rate (FRR) (staff convenience). By moving from a static model to your 15–20 signature Master Vector and Active Learning approach, you can push accuracy from your current 87% to a professional range of 94%–98%. 
ScienceDirect.com
ScienceDirect.com
 +4
1. Projected Accuracy Breakdown
At this scale, your accuracy will vary based on the "difficulty" of the signature and the presence of forgeries. 
IEEE
IEEE
 +1
Genuine Verification: 96%–98.5%. Using a Master Vector (the mathematical average of your 20 baseline samples) significantly reduces errors caused by daily minor variations in handwriting.
Random Forgeries: ~99%. The system is extremely good at stopping someone who just scribbles a random name or tries a casual guess.
Skilled Forgeries: 90%–93%. This is the most challenging area. Forgers who have seen the original signature can be difficult to catch with image-only data, but your active gallery helps by identifying "unnatural" perfection that doesn't match the user's historical variations. 
ScienceDirect.com
ScienceDirect.com
 +7
False Acceptance Rate (FAR) and False Rejection Rate (FRR ...
False rejection rate and false acceptance rate of a ...
GitHub - medAli-ai/Siamese-signature-verification-with ...
2. Why your accuracy will improve with 50k users
The Power of the Centroid: Storing an average Master Vector filters out the "noise" from your 20 baseline samples, creating a much more stable reference than a single "clean" image.
Reduced Intra-class Variability: By continuously updating the gallery through Active Learning, you "follow" a doctor's signature as it naturally evolves over years, keeping the False Rejection Rate (FRR) low.
One-Shot Advantage: Using a Siamese Network trained on general datasets (like CEDAR or GPDS) ensures the system is robust even for users it has only seen a few times. 
ScienceDirect.com
ScienceDirect.com
 +2
3. Managing the "Scale Risks"
With 50,000 users, even a 1% error rate means 500 people per day might face issues. To prevent this: 
Plurilock
Plurilock
Set Tiered Thresholds: Require 95%+ similarity for the system to "learn" a new signature into the gallery, but accept 85%+ just to mark attendance. This prevents "Template Poisoning" where a bad signature ruins the baseline.
Regular Benchmarking: Continuously monitor your Equal Error Rate (EER)—the point where FAR and FRR are balanced—to ensure the system isn't becoming too "strict" or too "loose" over time. 
ScienceDirect.com
ScienceDirect.com
 +3
Would you like to see how to adjust your similarity "thresholds" to specifically target a 99% success rate for your staff?
