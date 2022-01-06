import glob
from PIL import Image
from natsort import natsorted


def create_gif(img_path_list: list, out_filename: str = "animation.gif"):
    """
    複数の画像からGIFアニメーションを作成
    """
    imgs = []

    for i in range(len(img_path_list)):
        img = Image.open(img_path_list[i])
        imgs.append(img)

    imgs[0].save(
        out_filename,
        save_all=True,
        append_images=imgs[1:],
        optimize=False,
        duration=1000,
        loop=0,
    )


if __name__ == "__main__":
    img_dir = (
        "/home/kengoaraki/Project/DA/SSDA_MME_Saito_WSI/output/result/weightedMME/03_G144/"
    )
    # img_path_list = natsorted(glob.glob(img_dir + "*_pca.png"))
    # img_path_list = [img_dir + "weightedMME_MF0012_to_MF0003_cl[0, 1, 2]_cv0_iter1_03_G144_pca.png"]
    # img_path_list += [
    #     img_dir
    #     + f"weightedMME_MF0012_to_MF0003_cl[0, 1, 2]_cv0_iter{iteration}_03_G144_pca.png"
    #     for iteration in range(1000, 24001, 1000)
    # ]

    out_filename = img_dir + "weightedMME_MF0012_to_MF0003_cv0_iter1_to_iter240000_each1000_03_G144_pca.gif"

    create_gif(img_path_list=img_path_list, out_filename=out_filename)
