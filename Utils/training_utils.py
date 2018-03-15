import os
      #rt out to utilities and save all models in the same path
def loadModel(name, weights = False):
    json_file = open('%s.json' % name, 'r')
    loadedmodeljson = json_file.read()
    json_file.close()
    
    model = model_from_json(loadedmodeljson)
    
    #load weights into new model
    if weights == True:
        model.load_weights('%s.h5' % name)
    #print model.summary()
    print("Loaded model from disk")
    return model

def saveModel(model, name = "neural network", outputdir='./tmp'):
    if not os.path.isdir(outputdir) :os.mkdir(outputdir)
    model_name = name
    model.summary()
    model.save_weights(os.path.join(outputdir,'%s.h5' % model_name), overwrite=True)
    model_json = model.to_json()
    with open(os.path.join(outputdir,"%s.json" % model_name), "w") as json_file:
        json_file.write(model_json)
        
def saveLosses(hist, name="neural network"):    
    loss = np.array(hist.history['loss'])
    valoss = np.array(hist.history['val_loss'])
    f = h5py.File("%s_losses.h5" % name, "w")
    f.create_dataset('loss', data=loss)
    f.create_dataset('val_loss', data=valoss)
    f.close()
def show_losses( histories ):
    plt.figure(figsize=(5,5))
    #plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    colors = []
    do_acc = False
    
    for label,loss in histories:
        color = tuple(np.random.random(3))
        colors.append(color)
        l = label
        vl= label + " validation"
        
        if 'acc' in loss.history:
            l += ' (acc %2.4f)' % (loss.history['acc'][-1])
            do_acc = True
            
        if 'val_acc' in loss.history:
            vl += ' (val acc %2.4f)' % (loss.history['val_acc'][-1])
            do_acc = True
        plt.plot(loss.history['loss'], label=l, color=color)
        
        if 'val_loss' in loss.history:
            plt.plot(loss.history['val_loss'], lw=2, ls='dashed', label=vl, color=color)

    plt.legend()
    plt.yscale('log')
    plt.show()
    
    if not do_acc: return
    
    plt.figure(figsize=(5,5))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    for i,(label,loss) in enumerate(histories):
        color = colors[i]
        if 'acc' in loss.history:
            plt.plot(loss.history['acc'], lw=2, label=label+" accuracy", color=color)
            
        if 'val_acc' in loss.history:
            plt.plot(loss.history['val_acc'], lw=2, ls='dashed', label=label+" validation accuracy", color=color)
    
    plt.legend(loc='lower right')
    plt.show()
