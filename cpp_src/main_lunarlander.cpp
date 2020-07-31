#include <stdio.h>
#include <string>
#include <gtk/gtk.h>
#include "torch/torch.h"
using namespace torch;
#include "gamecontent.h"
#include "actorcritic.h"
#include "categorical.h"
#include "ppo.h"
#include "flowcontrol.h"

static FlowControl* s_mainprocess;

static void BtnHello( GtkWidget* widget, gpointer data )
{
    g_print("Hello World\n");

    s_mainprocess->Start();
    //debug test
   	s_mainprocess->TrainingTest();
}

static gboolean delete_event(GtkWidget *widget,
                         	 GdkEvent  *event,
							 gpointer   data)
{
	g_print("Quit Project\n");
	delete s_mainprocess;

	gtk_main_quit();
	return TRUE;
}

int main(int argc, char *argv[])
{
	GtkWidget*	window;
	GtkWidget*	button;

	s_mainprocess = new FlowControl();

	gtk_init (&argc, &argv);
	window = gtk_window_new (GTK_WINDOW_TOPLEVEL);
	gtk_widget_set_usize(window, 320, 240);
	g_signal_connect (window, "delete-event", G_CALLBACK (delete_event), NULL);
	gtk_container_set_border_width (GTK_CONTAINER (window), 75);

	button = gtk_button_new_with_label ("Press Start");
	g_signal_connect (button, "clicked", G_CALLBACK (BtnHello), NULL);

	gtk_container_add (GTK_CONTAINER (window), button);
	gtk_widget_show_all(window);

	gtk_main ();

	return 0;
}
