import os
import openslide
import pathlib

import numpy as np
from PIL import Image
import cv2
from scipy import stats


class OpenSlideWSI(openslide.OpenSlide):

    def __init__(self, filename, bg_mask_dir=None, semantic_mask_dir=None, dfb_dir=None):
        super().__init__(filename)
        p_filename = pathlib.Path(filename)
        self.wsi_name = str(p_filename.stem)
        self.wsi_obj_format = '{wsi_name}_{obj_idx:03d}'.format

        if bg_mask_dir is not None:
            self.bg_mask_dir = bg_mask_dir
            self.filename_bg_mask = self.bg_mask_dir \
                + self.wsi_name + "_mask_level05_bg.tif"
        if semantic_mask_dir is not None:
            self.semantic_mask_dir = semantic_mask_dir
            self.filename_semantic_mask = self.semantic_mask_dir \
                + self.wsi_name + "_overlaid.tif"
        if dfb_dir is not None:
            self.dfb_dir = dfb_dir
            self.filename_dfb = self.dfb_dir \
                + self.wsi_name + "_dfb.png"

    # 条件を満たさなければNoneを返す
    def _get_output_dir(self, s_p, output_main_dir, obj_name, th=1):
        output_dir = None
        s_p_mode = stats.mode(
            s_p, axis=None
        )  # s_p_mode[0]:s_pの最頻値, s_p_mode[1]:s_pの最頻値の個数
        # 背景領域が多いパッチは除外
        if int(s_p_mode[0]) != 255:
            # semanticパッチのクラス最頻値のピクセル数の割合と閾値を比較
            if float(s_p_mode[1] / (s_p.shape[0] * s_p.shape[1])) >= th:
                output_dir = f"{output_main_dir}{int(s_p_mode[0])}/{obj_name}/"
                os.makedirs(output_dir) if os.path.isdir(output_dir) is False else None
        return output_dir

    def _make_output_dir(self, output_main_dir, obj_name):
        output_dir = f"{output_main_dir}/{obj_name}/"
        os.makedirs(output_dir) if os.path.isdir(output_dir) is False else None
        return output_dir

    def _getBoundingBox(self, test_dir=None):
        bg_mask = cv2.imread(self.filename_bg_mask, cv2.IMREAD_GRAYSCALE)
        bg_mask_inv = np.zeros((bg_mask.shape), dtype=np.uint8)
        bg_mask_inv[bg_mask == 0] = 255

        nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(bg_mask_inv)

        bb_list = []
        obj_idx = 0
        for i in range(1, nlabels):
            x, y, w, h, obj_size = stats[i]
            wsi_obj_name = self.wsi_obj_format(wsi_name=self.wsi_name, obj_idx=obj_idx)

            if obj_size >= 10000:
                cv2.rectangle(bg_mask_inv, (x, y), (x + w, y + h), 255, cv2.LINE_4)
                stat = stats[i]
                bb_list.append({'x': stat[0], 'y': stat[1], 'w': stat[2],
                                'h': stat[3], 'name': wsi_obj_name})
                obj_idx += 1
            else:
                print(f"[Exclude] {wsi_obj_name} (size: {obj_size})")
                bg_mask_inv[labels == i] = 0

        if (test_dir is not None) and (obj_idx >= 1):
            cv2.imwrite(test_dir + wsi_obj_name + ".png", bg_mask_inv)

        return bb_list

    # split a bounding-box area of wsi to patch images
    def bb_to_patch(
        self,
        default_level,
        level,
        size,
        stride,
        bb,
        output_main_dir,
        contours_th=0.5,
    ):  # size=(width, height)
        assert isinstance(bb, dict), "bb(bounding-box) must be dict type"
        # bounding_box
        bx, by, bw, bh = bb['x'], bb['y'], bb['w'], bb['h']
        obj_name = bb['name']

        # bbをlevel[0]用に変換（bbはbg_mask_level05における座標のため）
        bx_wsi = bb['x'] * (2 ** (default_level - level))
        by_wsi = bb['y'] * (2 ** (default_level - level))
        bw_wsi = bb['w'] * (2 ** (default_level - level))
        bh_wsi = bb['h'] * (2 ** (default_level - level))

        width = self.level_dimensions[level][0]
        height = self.level_dimensions[level][1]
        row_max = int((bw_wsi - size[0]) / stride + 1)
        column_max = int((bh_wsi - size[1]) / stride + 1)

        stride_rate = stride / 2 ** (default_level - level)
        width_rate = size[0] / 2 ** (default_level - level)
        height_rate = size[1] / 2 ** (default_level - level)

        assert self.filename_semantic_mask is not None, "Should set filename_semantic_mask"
        semantic_mask = Image.open(self.filename_semantic_mask)
        semantic_mask_np = np.array(semantic_mask)
        assert self.filename_bg_mask is not None, "Should set filename_bg_mask"
        bg_mask_np = np.array(Image.open(self.filename_bg_mask))

        cnt = 0
        for column in range(column_max):
            for row in range(row_max):
                i = int(bx_wsi + (row * stride * (2 ** level)))
                j = int(by_wsi + (column * stride * (2 ** level)))

                mask_base_idx = {'row': int(bx + (row * stride_rate)),
                                 'col': int(by + (column * stride_rate))}

                # width_rate×height_rateの領域(背景領域のマスク画像)の画素値が0の画素数で比較
                if (
                    len(
                        np.where(
                            bg_mask_np[
                                mask_base_idx['col']:int(mask_base_idx['col'] + height_rate),
                                mask_base_idx['row']: int(mask_base_idx['row'] + width_rate),
                            ]
                            == 0
                        )[0]
                    )
                    >= contours_th * height_rate * width_rate
                ):
                    # width_rate×height_rateの領域(semanticマスク)の画素値が255以外(背景以外)の画素数で比較
                    if (
                        len(
                            np.where(
                                semantic_mask_np[
                                    mask_base_idx['col']:int(mask_base_idx['col'] + height_rate),
                                    mask_base_idx['row']:int(mask_base_idx['row'] + width_rate),
                                ]
                                != 255
                            )[0]
                        )
                        >= contours_th * height_rate * width_rate
                    ):

                        s_p = semantic_mask_np[
                            mask_base_idx['col']:int(mask_base_idx['col'] + height_rate),
                            mask_base_idx['row']:int(mask_base_idx['row'] + width_rate)
                        ]

                        output_dir = self._get_output_dir(
                            s_p, output_main_dir, obj_name)

                        if output_dir is not None:
                            # self.read_region((i, j), level, size).save(
                            #     output_dir
                            #     + str(level)
                            #     + "_"
                            #     + str(cnt).zfill(10)
                            #     + ".png"
                            # )

                            # FIXME: rewrite dfb crop
                            if self.filename_dfb is not None:
                                dfb = Image.open(self.filename_dfb)
                                output_dfb_dir = output_dir.replace("/mnt2/", "/mnt3/")
                                os.makedirs(output_dfb_dir) if os.path.isdir(output_dfb_dir) is False else None
                                assert os.path.exists(output_dfb_dir), f"{output_dfb_dir} does not exists"
                                dfb.crop(
                                    (
                                        mask_base_idx['row'],
                                        mask_base_idx['col'],
                                        int(mask_base_idx['row'] + width_rate),
                                        int(mask_base_idx['col'] + height_rate),
                                    )
                                ).resize((size[0], size[1])).save(
                                    output_dfb_dir
                                    + str(level)
                                    + "_"
                                    + str(cnt).zfill(10)
                                    + ".png"
                                )

                                # --- output mean of dfb patch --- #
                                # dfb_num = np.mean(
                                #     np.asarray(
                                #         dfb.crop(
                                #             (
                                #                 mask_base_idx['row'],
                                #                 mask_base_idx['col'],
                                #                 int(mask_base_idx['row'] + width_rate),
                                #                 int(mask_base_idx['col'] + height_rate),
                                #             )
                                #         )
                                #     )
                                # ).astype(np.uint8)

                                # dfb_patch = Image.fromarray(
                                #     np.full((size[0], size[1]), dfb_num)
                                # )

                                # dfb_patch.save(
                                #     output_dfb_dir
                                #     + str(level)
                                #     + "_"
                                #     + str(cnt).zfill(10)
                                #     + ".png"
                                # )
                                # -------------------------------- #

                            cnt = cnt + 1

    # split a full-size image to patch images
    # this patch is used for making prediction-map
    def image_to_patch(
        self,
        default_level,
        level,
        size,
        stride,
        output_main_dir,
        obj_name,
        cnt=0,
    ):  # size=(width,height)

        width = self.level_dimensions[level][0]
        height = self.level_dimensions[level][1]
        row_max = int((width - size[0]) / stride + 1)
        column_max = int((height - size[1]) / stride + 1)

        stride_rate = stride / 2 ** (default_level - level)
        width_rate = size[0] / 2 ** (default_level - level)
        height_rate = size[1] / 2 ** (default_level - level)

        # bg_maskとsemantic_maskを入力に与えなかった場合
        for column in range(column_max):
            for row in range(row_max):
                i = int(row * stride * (2 ** level))
                j = int(column * stride * (2 ** level))

                output_dir = self._make_output_dir(output_main_dir, obj_name)

                self.read_region((i, j), level, size).save(
                    output_dir
                    + str(level)
                    + "_"
                    + str(cnt).zfill(10)
                    + ".png"
                )

                if self.filename_dfb is not None:
                    dfb = Image.open(self.filename_dfb)
                    # FIXME: 定性評価画像用のpatchのdirであることに注意！
                    output_dfb_dir = output_dir.replace("/mnt4(ForPredmap)/", "/mnt4(ForPredmap_DFB_Mean)/")
                    os.makedirs(output_dfb_dir) if os.path.isdir(output_dfb_dir) is False else None
                    assert os.path.exists(output_dfb_dir), f"{output_dfb_dir} does not exists"

                    # dfb.crop(
                    #     (
                    #         row * stride_rate,
                    #         column * stride_rate,
                    #         row * stride_rate + width_rate,
                    #         column * stride_rate + height_rate,
                    #     )
                    # ).resize((size[0], size[1])).save(
                    #     output_dfb_dir
                    #     + str(level)
                    #     + "_"
                    #     + str(cnt).zfill(10)
                    #     + ".png"
                    # )

                    # --- output mean of dfb patch --- #
                    dfb_num = np.mean(
                        np.asarray(
                            dfb.crop(
                                (
                                    row * stride_rate,
                                    column * stride_rate,
                                    row * stride_rate + width_rate,
                                    column * stride_rate + height_rate,
                                )
                            )
                        )
                    ).astype(np.uint8)

                    dfb_patch = Image.fromarray(
                        np.full((size[0], size[1]), dfb_num)
                    )

                    dfb_patch.save(
                        output_dfb_dir
                        + str(level)
                        + "_"
                        + str(cnt).zfill(10)
                        + ".png"
                    )
                    # -------------------------------- #

                cnt = cnt + 1
        return cnt

    # merge patch images to a full size image
    def patch_to_image(
        self,
        resized_size,
        level,
        size,
        stride,
        input_dir,
        output_dir,
        output_name,
        suffix=None,
        cnt=0,
    ):

        width = self.level_dimensions[level][0]
        height = self.level_dimensions[level][1]
        row_max = int((width - size[0]) / stride + 1)
        column_max = int((height - size[1]) / stride + 1)

        canvas = Image.new(
            "RGB",
            (resized_size[0] * row_max, resized_size[1] * column_max),
            (255, 255, 255),
        )

        for column in range(column_max):
            for row in range(row_max):
                if suffix is None:
                    img = Image.open(
                        input_dir + str(level) + "_" + str(cnt).zfill(10) + ".png", "r"
                    ).resize((resized_size[0], resized_size[1]))
                else:
                    img = Image.open(
                        input_dir + str(level) + "_" + str(cnt).zfill(10) + str(suffix) + ".png", "r"
                    ).resize((resized_size[0], resized_size[1]))
                canvas.paste(img, (row * resized_size[0], column * resized_size[1]))
                cnt = cnt + 1
        if suffix is None:
            canvas.save(output_dir + output_name + ".png", "PNG", quality=100)
        else:
            canvas.save(output_dir + output_name + str(suffix) + ".png", "PNG", quality=100)

        return cnt

    # # split a full-size image to patch images
    # def image_to_patch(
    #     self,
    #     default_level,
    #     level,
    #     size,
    #     stride,
    #     output_path_patch,
    #     cnt,
    #     mode='train',
    #     output_path_semantic_mask=None,
    #     contours_th=1,
    # ):  # size=(width,height)

    #     width = self.level_dimensions[level[0]][0]
    #     height = self.level_dimensions[level[0]][1]
    #     row_max = int((width - size[0]) / stride + 1)
    #     column_max = int((height - size[1]) / stride + 1)

    #     stride_rate = stride / 2 ** (default_level - level[0])
    #     width_rate = size[0] / 2 ** (default_level - level[0])
    #     height_rate = size[1] / 2 ** (default_level - level[0])

    #     if mode == 'train':  # generate training patch

    #         semantic_mask = Image.open(self.filename_semantic_mask)
    #         semantic_mask_np = np.array(semantic_mask)
    #         bg_mask_np = np.array(Image.open(self.filename_bg_mask))
    #         for column in range(column_max):
    #             for row in range(row_max):
    #                 i = int(row * stride * (2 ** level[0]))
    #                 j = int(column * stride * (2 ** level[0]))

    #                 # width_rate×height_rateの領域(背景領域のマスク画像)の画素値が0の画素数で比較
    #                 if (
    #                     len(
    #                         np.where(
    #                             bg_mask_np[
    #                                 int(column * stride_rate) : int(
    #                                     column * stride_rate + height_rate
    #                                 ),
    #                                 int(row * stride_rate) : int(
    #                                     row * stride_rate + width_rate
    #                                 ),
    #                             ]
    #                             == 0
    #                         )[0]
    #                     )
    #                     >= contours_th * height_rate * width_rate
    #                 ):
    #                     # width_rate×height_rateの領域(semanticマスク)の画素値が255以外(背景以外)の画素数で比較
    #                     if (
    #                         len(
    #                             np.where(
    #                                 semantic_mask_np[
    #                                     int(column * stride_rate) : int(
    #                                         column * stride_rate + height_rate
    #                                     ),
    #                                     int(row * stride_rate) : int(
    #                                         row * stride_rate + width_rate
    #                                     ),
    #                                 ]
    #                                 != 255
    #                             )[0]
    #                         )
    #                         >= contours_th * height_rate * width_rate
    #                     ):
    #                         self.read_region((i, j), level[0], size).save(
    #                             output_path_patch
    #                             + str(level[0])
    #                             + "_"
    #                             + str(cnt).zfill(10)
    #                             + ".png"
    #                         )
    #                         semantic_mask.crop(
    #                             (
    #                                 row * stride_rate,
    #                                 column * stride_rate,
    #                                 row * stride_rate + width_rate,
    #                                 column * stride_rate + height_rate,
    #                             )
    #                         ).resize((size[0], size[1])).save(
    #                             output_path_semantic_mask
    #                             + str(level[0])
    #                             + "_"
    #                             + str(cnt).zfill(10)
    #                             + ".png"
    #                         )

    #                         cnt = cnt + 1
    #                         # if cnt >= 30:
    #                         #     exit(1)

    #     # bg_maskとsemantic_maskを入力に与えなかった場合
    #     elif mode == 'test':  # generate test patch
    #         for column in range(column_max):
    #             for row in range(row_max):
    #                 i = int(row * stride * (2 ** level[0]))
    #                 j = int(column * stride * (2 ** level[0]))

    #                 self.read_region((i, j), level[0], size).save(
    #                     output_path_patch
    #                     + str(level[0])
    #                     + "_"
    #                     + str(cnt).zfill(10)
    #                     + ".png"
    #                 )
    #                 cnt = cnt + 1
    #     else:
    #         sys.exit("In image_to_patch, setting error occured.")
    #     return cnt


def main():
    parent_dir = "/mnt/ssdsub1/DFBConv_strage/mnt1/MF0012/"
    p_parent_dir = pathlib.Path(parent_dir)
    output_main_dir = parent_dir.replace("mnt1/", "mnt2/")

    # ------------------ #
    DEFAULT_LEVEL = 5
    LEVEL = 0
    SIZE = (256, 256)
    STRIDE = 256
    CONTOURS_TH = 1
    CLASSES = [0, 1, 2, 3, 4]
    # ------------------ #

    wsi_list = sorted([wsi_path for wsi_path in (p_parent_dir / "origin/").iterdir()])

    for wsi_path in wsi_list:
        wsi_path = str(wsi_path)
        bg_mask_dir = parent_dir + "mask_bg/"
        test_dir = parent_dir + "test/"
        semantic_mask_dir = parent_dir + \
            f"mask_cancergrade_gray/overlaid_{CLASSES}/"
        dfb_dir = parent_dir + "dfb/"

        wsi = OpenSlideWSI(
            wsi_path,
            bg_mask_dir=bg_mask_dir,
            semantic_mask_dir=semantic_mask_dir,
            dfb_dir=dfb_dir)

        print("==== {} ====".format(wsi.wsi_name))
        bb_list = wsi._getBoundingBox()

        for bb in bb_list:
            print(bb['name'])

            wsi.bb_to_patch(
                DEFAULT_LEVEL,
                LEVEL,
                SIZE,
                STRIDE,
                bb,
                output_main_dir,
                CONTOURS_TH,
            )


# 予測画像用のパッチ切り取り
def main_for_predmap():
    parent_dir = "/mnt/ssdsub1/DFBConv_strage/mnt1/MF0012/"
    p_parent_dir = pathlib.Path(parent_dir)
    output_main_dir = parent_dir.replace("mnt1/", "mnt4(ForPredmap)/")

    # ------------------ #
    DEFAULT_LEVEL = 5
    LEVEL = 0
    SIZE = (256, 256)
    STRIDE = 256
    # ------------------ #

    wsi_list = sorted([wsi_path for wsi_path in (p_parent_dir / "origin/").iterdir()])

    for wsi_path in wsi_list:
        wsi_path = str(wsi_path)
        dfb_dir = parent_dir + "dfb/"

        wsi = OpenSlideWSI(
            wsi_path,
            bg_mask_dir=None,
            semantic_mask_dir=None,
            dfb_dir=dfb_dir)

        print("==== {} ====".format(wsi.wsi_name))
        wsi.image_to_patch(DEFAULT_LEVEL, LEVEL, SIZE, STRIDE, output_main_dir, wsi.wsi_name)


if __name__ == "__main__":
    # main()
    main_for_predmap()
