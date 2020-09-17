    # TF dataset creation start
    def ds_gen():
        while True:
            indrand = np.random.randint(len(list_train_PAN))
            filename1 = list_train_PAN[indrand]
            filename2 = list_train_MS[indrand]

            imlistPAN = [filename1]
            imlistMS = [filename2]

            hrpan_bat, hrms_bat, lrpan_bat, lrms_bat = helper.im2subim(NUM_SUB_PER_IM, SUBIM_SIZE_LR_MS, imlistPAN, imlistMS, SCALE)
            hrpan_bat = np.array(hrpan_bat, dtype=np.float32) / IMG_DIV
            hrms_bat = np.array(hrms_bat, dtype=np.float32) / IMG_DIV
            lrpan_bat = np.array(lrpan_bat, dtype=np.float32) / IMG_DIV
            lrms_bat = np.array(lrms_bat, dtype=np.float32) / IMG_DIV

            yield hrpan_bat, hrms_bat, lrpan_bat, lrms_bat


    def tf_ds_gen():
        dataset0 = (tf.data.Dataset.from_generator(ds_gen, (tf.float32, tf.float32, tf.float32, tf.float32))
                    )
        return dataset0

    dataset = (tf.data.Dataset.range(100).repeat()
               .apply(
        tf.data.experimental.parallel_interleave(lambda filename: tf_ds_gen(), cycle_length=int(N_CPU), sloppy=False))
               .apply(tf.data.experimental.unbatch())
               .shuffle(buffer_size=int(NUM_SUB_PER_IM * 10))
               .batch(int(BATCH_SIZE))
               )

  # Extract subimages from each training image (number of subimages per image = NUM_SUB_PER_IM)
def im2subim(NUM_SUB_PER_IM, SUBIM_SIZE_LR_MS, imlistP, imlistM, SCALE):
    cropListHRPAN = list()
    cropListHRMS = list()
    cropListLRPAN = list()
    cropListLRMS = list()
    for imIndex in range(len(imlistP)):
        impan = tifffile.imread(imlistP[imIndex])
        impan = impan[:, :, np.newaxis]
        imms = tifffile.imread(imlistM[imIndex])

        impan = impan.astype(np.float32)
        imms = imms.astype(np.float32)

        szpan = impan.shape
        szpan = np.array(szpan, dtype=np.float32)
        szms = np.floor(szpan / (4 * 5)) * 5
        szpan = szms * 4
        szpan = szpan.astype(np.int)
        szms = szms.astype(np.int)

        impan = impan[:szpan[0], :szpan[1], :]
        imms = imms[:szms[0], :szms[1], :]

        szlrms = SUBIM_SIZE_LR_MS
        szhrms = int(np.floor(szlrms * SCALE))
        szlrpan = szlrms * 4
        szhrpan = szhrms * 4

        ih, iw, _ = imms.shapems_data
        nw, nh = iw - szhrms, ih - szhrms

        # iml = rescale(imh, 1.0 / SCALE, anti_aliasing=True, multichannel=True)

        for subImIndex in range(NUM_SUB_PER_IM):
            if nw == 0:
                indw = 0
            else:
                indw = random.randint(0, nw)
            if nh == 0:
                indh = 0
            else:
                indh = random.randint(0, nh)

            impan_hr = impan[indh * 4:(indh + szhrms) * 4, indw * 4:(indw + szhrms) * 4, :]
            imms_hr = imms[indh:indh + szhrms, indw:indw + szhrms, :]

            impan_lr = rescale(impan_hr, 1.0 / SCALE, anti_aliasing=True, multichannel=True)
            imms_lr = rescale(imms_hr, 1.0 / SCALE, anti_aliasing=True, multichannel=True)

            cropListHRPAN.append(impan_hr)
            cropListHRMS.append(imms_hr)
            cropListLRPAN.append(impan_lr)
            cropListLRMS.append(imms_lr)

    batch_temp = [[cropListHRPAN[cropImIndex], cropListHRMS[cropImIndex], cropListLRPAN[cropImIndex], cropListLRMS[cropImIndex]] for cropImIndex in range(len(cropListHRPAN))]

    hrpan, hrms, lrpan, lrms = zip(*batch_temp)

    return hrpan, hrms, lrpan, lrms


iterator = dataset.make_one_shot_iterator()
