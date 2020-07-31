package com.example.ppo;

import android.app.Activity;
import android.os.Bundle;

public class PPOViewActivity extends Activity
{
    public static native void nativeOnStart();
    public static native void nativeOnPause();
    public static native void nativeOnResume();
    public static native void nativeOnStop();

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        System.loadLibrary("ppo");

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_card_view);

        nativeOnStart();
        nativeOnPause();
        nativeOnResume();

        if (savedInstanceState == null)
        {
            getFragmentManager().
                    beginTransaction().
                    add(R.id.container, PPOViewFragment.newInstance()).
                    commit();
        }

        nativeOnPause();
        nativeOnResume();
        nativeOnStop();
    }
}
