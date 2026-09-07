"""
Coin Segmentation Pipeline (Workshop: Object vs Background Segmentation)
Responsibility: กานต์นิธิ — Algorithm & Morphology (ขั้นตอน 3-4)

โครงสร้างข้อมูลที่คาดหวัง (มาจากงานของเมธา ขั้นตอน 1-2):
    dataset/
        images/   -> ภาพเหรียญต้นฉบับ (.jpg / .png)
        masks/    -> ground truth mask (ถ้ามี, ไม่ใช้ในไฟล์นี้ แต่ใช้ตอน evaluation)

สิ่งที่สคริปต์นี้ทำ:
    1. โหลดภาพ
    2. ทำ Segmentation ด้วย 2 วิธี: Otsu Thresholding (grayscale) และ HSV Color Thresholding
    3. ใช้ Morphology (Opening/Closing) ทำความสะอาดผลลัพธ์
    4. บันทึกภาพเปรียบเทียบ before/after morphology ไว้ใน output/
    5. บันทึก predicted mask สุดท้ายไว้ใน output/masks/ สำหรับให้ญานพัทธ์ (Evaluation) ใช้ต่อ
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIG - แก้ path ตรงนี้ให้ตรงกับโฟลเดอร์จริงของทีม
# ---------------------------------------------------------------------------
INPUT_DIR = "dataset/images"
OUTPUT_MASK_DIR = "output/masks"
OUTPUT_COMPARE_DIR = "output/compare"

# ค่าเริ่มต้นของ kernel morphology (ปรับทดลองได้)
MORPH_KERNEL_SIZE = 5
MORPH_ITERATIONS = 2


# ---------------------------------------------------------------------------
# STEP 3: SEGMENTATION ALGORITHMS
# ---------------------------------------------------------------------------

def segment_otsu(img_bgr, assume_object_minority=True):
    """
    วิธีที่ 1: Global thresholding แบบอัตโนมัติด้วย Otsu's method
    เหมาะกับกรณีเหรียญ (โลหะ, สีเทา/เงิน/ทอง) ตัดกับพื้นหลังชัดเจนด้านความสว่าง

    ปัญหาที่พบบ่อย: Otsu ไม่รู้ว่าฝั่งไหนคือ "วัตถุ" ฝั่งไหนคือ "พื้นหลัง"
    -> เราจึงเช็คอัตโนมัติ: ปกติพื้นที่เหรียญในภาพควรเป็น "ส่วนน้อย" ของพื้นที่ทั้งหมด
       ถ้า mask ที่ได้มีพิกเซลขาว (positive) เกินครึ่งภาพ แสดงว่ากลับขั้ว ให้ invert คืน
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # เบลอเล็กน้อยลด noise ก่อน threshold ช่วยให้ Otsu เสถียรขึ้น
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if assume_object_minority:
        white_ratio = np.count_nonzero(mask) / mask.size
        if white_ratio > 0.5:
            mask = cv2.bitwise_not(mask)

    return mask, gray  # gray ใช้เป็น "score map" สำหรับ ROC ในขั้นตอน evaluation


def segment_hsv(img_bgr, lower_hsv, upper_hsv):
    """
    วิธีที่ 2: Color-based thresholding ใน HSV space
    ใช้เมื่อเหรียญมีโทนสีต่างจากพื้นหลังชัด (เช่น เหรียญทองแดง/ทองเหลือง บนพื้นผ้าสีเข้ม)
    ต้องปรับ lower_hsv / upper_hsv ให้เข้ากับสีเหรียญและพื้นหลังจริงของแต่ละภาพ
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    return mask


def segment_adaptive(img_bgr):
    """
    วิธีเสริม: Adaptive thresholding — มีประโยชน์เมื่อแสงในภาพไม่สม่ำเสมอ
    (เช่น เงาตกกระทบบางส่วนของภาพ)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    mask = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=25, C=5
    )
    return mask


# ---------------------------------------------------------------------------
# STEP 4: MORPHOLOGY CLEANUP
# ---------------------------------------------------------------------------

def apply_morphology(mask, kernel_size=MORPH_KERNEL_SIZE, iterations=MORPH_ITERATIONS):
    """
    Opening  (erode -> dilate) : ลบจุด noise เล็กๆ รอบเหรียญ
    Closing  (dilate -> erode) : ปิดรูโหว่/แสงสะท้อนตรงกลางเหรียญ
    ทำ Opening ก่อน แล้วค่อย Closing เพื่อไม่ให้รูโหว่ถูกนับเป็น noise ไปด้วย
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return closed


def keep_largest_contours(mask, min_area_ratio=0.001):
    """
    Post-processing เสริม: ลบ blob เล็กๆ ที่หลุดรอดจาก morphology
    (เช่น เศษแสงสะท้อน/ฝุ่นที่ยังถูกจัดเป็น positive)
    เก็บเฉพาะ contour ที่มีพื้นที่ใหญ่กว่า threshold ที่กำหนด (สัดส่วนของพื้นที่ภาพทั้งหมด)
    """
    h, w = mask.shape
    min_area = h * w * min_area_ratio
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    clean_mask = np.zeros_like(mask)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(clean_mask, [c], -1, 255, thickness=cv2.FILLED)

    return clean_mask


# ---------------------------------------------------------------------------
# VISUALIZATION: before / after morphology comparison
# ---------------------------------------------------------------------------

def save_comparison(img_bgr, mask_before, mask_after, out_path, title=""):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")

    axes[1].imshow(mask_before, cmap="gray")
    axes[1].set_title("Before Morphology\n(raw threshold)")

    axes[2].imshow(mask_after, cmap="gray")
    axes[2].set_title("After Morphology\n(open + close)")

    overlay = img_bgr.copy()
    overlay[mask_after > 0] = (0, 255, 0)
    blended = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
    axes[3].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Final Overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def process_image(path, method="otsu", hsv_range=None):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"อ่านภาพไม่ได้: {path}")

    if method == "otsu":
        mask_before, score_map = segment_otsu(img)
    elif method == "hsv":
        assert hsv_range is not None, "ต้องระบุ hsv_range = (lower, upper) เมื่อใช้ method='hsv'"
        mask_before = segment_hsv(img, *hsv_range)
        score_map = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif method == "adaptive":
        mask_before = segment_adaptive(img)
        score_map = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("method ต้องเป็น 'otsu', 'hsv' หรือ 'adaptive'")

    mask_morph = apply_morphology(mask_before)
    mask_final = keep_largest_contours(mask_morph)

    return img, mask_before, mask_morph, mask_final, score_map


def run_pipeline(method="otsu", hsv_range=None):
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_COMPARE_DIR, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not image_files:
        print(f"ไม่พบภาพใน {INPUT_DIR} — ตรวจสอบ path หรือรอไฟล์จากเมธาก่อน")
        return

    for fname in image_files:
        path = os.path.join(INPUT_DIR, fname)
        name = os.path.splitext(fname)[0]

        img, mask_before, mask_morph, mask_final, score_map = process_image(
            path, method=method, hsv_range=hsv_range
        )

        # บันทึก mask สุดท้าย (นี่คือไฟล์ที่ญานพัทธ์จะเอาไปเทียบกับ ground truth)
        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, f"{name}_mask.png"), mask_final)

        # บันทึกภาพเปรียบเทียบ before/after morphology
        save_comparison(
            img, mask_before, mask_final,
            os.path.join(OUTPUT_COMPARE_DIR, f"{name}_compare.png"),
            title=f"{name} (method={method})"
        )

        print(f"processed: {fname}")

    print(f"\nเสร็จแล้ว! mask -> {OUTPUT_MASK_DIR}/, ภาพเปรียบเทียบ -> {OUTPUT_COMPARE_DIR}/")


if __name__ == "__main__":
    # ตัวอย่างการเรียกใช้:

    # วิธีที่ 1: Otsu (แนะนำเป็นค่าเริ่มต้นสำหรับเหรียญโลหะบนพื้นหลังทึบ)
    run_pipeline(method="otsu")

    # วิธีที่ 2: HSV (ถ้าเหรียญมีสีต่างจากพื้นหลังชัดเจน ปรับช่วงสีให้ตรงกับภาพจริง)
    # run_pipeline(method="hsv", hsv_range=([10, 50, 50], [30, 255, 255]))

    # วิธีที่ 3: Adaptive (ถ้าภาพมีปัญหาแสงไม่สม่ำเสมอ)
    # run_pipeline(method="adaptive")
