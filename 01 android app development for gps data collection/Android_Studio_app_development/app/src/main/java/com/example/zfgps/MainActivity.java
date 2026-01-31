package com.example.zfgps;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.location.Address;
import android.location.Criteria;
import android.location.Geocoder;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.widget.Button;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.location.Priority;
import com.google.android.gms.tasks.OnSuccessListener;
import com.opencsv.CSVWriter;

import java.io.File;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;

public class MainActivity extends AppCompatActivity  {
    public static final int defaultUpdateInterval = 1;
    public static final int fastestUpdateInterval = 1;
    public static final int PERMISSIONS_FINE_LOCATION = 99;
    TextView tv_lat, tv_lon, tv_altitude, tv_accuracy, tv_speed, tv_sensor, tv_updates, tv_address, tv_waypointCounts, tv_time;
    @SuppressLint("UseSwitchCompatOrMaterialCode")
    Switch sw_locationsupdates;
    @SuppressLint("UseSwitchCompatOrMaterialCode")
    Switch sw_gps;
    Button  btn_showWaypointList, btn_logData, btn_export;
    SimpleDateFormat simpleDateFormat;
    String DateAndTime;
    Calendar calender;
    MyGPSDatabase myGPSDatabase;
    //LocationManager locationManager;
    //boolean updateOn = false;
    Location currentLocation;
    List<Location> savedLocations;
    //Googles API for location provider
    FusedLocationProviderClient fusedLocationProviderClient;
    // Location Request
    LocationRequest locationRequest;
    LocationCallback locationCallBack;
    public Criteria criteria;
    public String bestProvider;


    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Log.d("MSG", "App is Starting");

        tv_lat = findViewById(R.id.tv_lat);
        tv_lon = findViewById(R.id.tv_lon);
        tv_altitude = findViewById(R.id.tv_altitude);
        tv_accuracy = findViewById(R.id.tv_accuracy);
        tv_speed = findViewById(R.id.tv_speed);
        tv_sensor = findViewById(R.id.tv_sensor);
        tv_updates = findViewById(R.id.tv_updates);
        tv_address = findViewById(R.id.tv_address);
        sw_locationsupdates = findViewById(R.id.sw_locationsupdates);
        sw_gps = findViewById(R.id.sw_gps);
        btn_showWaypointList = findViewById(R.id.btn_showWaypointList);
        tv_waypointCounts = findViewById(R.id.tv_countOfBreadCrumbs);
        tv_time = findViewById(R.id.tv_time);
        btn_logData = findViewById(R.id.btnLog);
        btn_export = findViewById(R.id.btnExport);

        // Creating the location request
        locationRequest = new LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 100 * defaultUpdateInterval)
                .setWaitForAccurateLocation(false)
                .setMinUpdateIntervalMillis(100 * fastestUpdateInterval)
                .build();
        Log.d("MYMSG", "Location Requester Build successsfully.");

        // Creating new database
        myGPSDatabase = new MyGPSDatabase(this);
        Log.d("MYMSG", "New DataBase Created");

        //Creating the Callback function which will execute the instructions after receiving the location signal.
        locationCallBack = new LocationCallback() {
            @Override
            public void onLocationResult(@NonNull LocationResult locationResult) {
                super.onLocationResult(locationResult);
                //save the location
                Log.d("MSG","location is being checked");
                Location location = locationResult.getLastLocation();
                if(location!=null){
                    Log.d("MSG","location is not null");

                    updateUIValues(location);

                    boolean result = myGPSDatabase.insertPositionsData(tv_time.getText().toString(),
                            tv_lat.getText().toString(),
                            tv_lon.getText().toString(),
                            tv_speed.getText().toString());
                    if (result){
                        Toast.makeText(MainActivity.this, "Success", Toast.LENGTH_LONG).show();
                    }
                    else{
                        Toast.makeText(MainActivity.this, "Fail", Toast.LENGTH_LONG).show();
                    }
                    //updateGPS();
                }
            }
        };

        btn_showWaypointList.setOnClickListener(v -> {
            Intent i = new Intent(MainActivity.this, ShowSavedLocationList.class);
            startActivity(i);

        });

        sw_gps.setOnClickListener(v -> {
            if (sw_gps.isChecked()) {
                locationRequest = new LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY,100*defaultUpdateInterval)
                        .setWaitForAccurateLocation(false)
                        .setMinUpdateIntervalMillis(100*fastestUpdateInterval)
                        .build();
                tv_sensor.setText("Using GPS Sensors");
            } else {
                locationRequest = new LocationRequest.Builder(Priority.PRIORITY_BALANCED_POWER_ACCURACY,100*defaultUpdateInterval)
                        .setWaitForAccurateLocation(false)
                        .setMinUpdateIntervalMillis(100*fastestUpdateInterval)
                        .build();
                tv_sensor.setText("Using Towers + WIFI");
            }
        });
        sw_locationsupdates.setOnClickListener(v -> {
            if (sw_locationsupdates.isChecked()) {
                updateGPS();
                startLocationUpdates();
            } else {
                stopLocationUpdates();
            }
        });
        btn_logData.setOnClickListener(v -> {
            // Get the gps location
            // Add the new location to the list

            MyApplication myApplication = (MyApplication)getApplicationContext();
            savedLocations = myApplication.getMyLocations();
            savedLocations.add(currentLocation);

            boolean result = myGPSDatabase.insertPositionsData(tv_time.getText().toString(),
                    tv_lat.getText().toString(),
                    tv_lon.getText().toString(),
                    tv_speed.getText().toString());
            if (result){
                Toast.makeText(MainActivity.this, "Success", Toast.LENGTH_SHORT).show();
            }
            else{
                Toast.makeText(MainActivity.this, "Fail", Toast.LENGTH_SHORT).show();
            }
        });
        btn_export.setOnClickListener(v -> {

            AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
            builder.setTitle("Export Warning")
                    .setMessage("Exporting data will delete local App Database. Do you want to proceed?")
                    .setPositiveButton("Yes", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialogInterface, int i) {
                            if(checkStoragePermission()){
                                exportCSV();
                                myGPSDatabase.clearDataFromDB();
                            }else{
                                requestStoragePermission();
                            }
                        }
                    })
                    .setNegativeButton("No", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialog, int i) {
                            dialog.cancel();
                        }
                    });
            builder.create().show();
        });
    }  //end of onCreate method

    private void exportCSV() {
        File folder = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_ALARMS+"/Data");
        if(!folder.exists()){
            folder.mkdirs();
        }

        calender = Calendar.getInstance();
        SimpleDateFormat sdf = new SimpleDateFormat("yyMMddHHmmssZZZ");
        String Datetime = sdf.format(calender.getTime());

        String csvFileName = Datetime+".csv";
        File file = new File(folder, csvFileName);
        String csvFilePath = folder+"/"+csvFileName;
        Log.d("EXP", csvFilePath);

        ArrayList<Position> dataList = new ArrayList<>();
        dataList = myGPSDatabase.getPositionData();
        try{
            file.createNewFile();
            FileWriter fw = new FileWriter(file);
            for(int i=0; i<dataList.size(); i++){
                fw.append("")
                        .append(String.valueOf(dataList.get(i).getId()))
                        .append(",")
                        .append(dataList.get(i).getTime())
                        .append(",")
                        .append(dataList.get(i).getLat())
                        .append(",")
                        .append(dataList.get(i).getLon())
                        .append(",")
                        .append(dataList.get(i).getSpeed())
                        .append("\n");
            }
            //fw.flush();
            fw.close();
            Toast.makeText(this, "Success", Toast.LENGTH_LONG).show();

        }catch(Exception e){
            Toast.makeText(this, e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }
    @SuppressLint("SetTextI18n")
    private void stopLocationUpdates() {
        tv_updates.setText("Location is tracking stopped");
        tv_time.setText("Not being tracked");
        tv_lat.setText("Not being tracked");
        tv_lon.setText("Not being tracked");
        tv_accuracy.setText("Not being tracked");
        tv_altitude.setText("Not being tracked");
        tv_speed.setText("Not being tracked");
        tv_sensor.setText("Not being tracked");
        tv_address.setText("Not being tracked");
        fusedLocationProviderClient.removeLocationUpdates(locationCallBack);
    }
    private void startLocationUpdates() {
        tv_updates.setText("Location is being tracked");
        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return;
        }
        fusedLocationProviderClient.requestLocationUpdates(locationRequest, locationCallBack, null);
        updateGPS();
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        Log.d("MSG","Entering the onREQUESTPermission Result ");
        if (requestCode == PERMISSIONS_FINE_LOCATION) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                updateGPS();
            } else {
                Toast.makeText(this, "This app requires permissions to be granted properly", Toast.LENGTH_LONG).show();
                finish();
            }
        }
    }
    private void updateGPS(){
        //Get Permissions from the user
        //Get Current location from fused client
        //Update the UI

        fusedLocationProviderClient = LocationServices.getFusedLocationProviderClient(MainActivity.this);
        if (checkLocationPermission()){
            Log.d("MSG","PERMISSION IS GIVEN");
            fusedLocationProviderClient.getLastLocation().addOnSuccessListener(this, new OnSuccessListener<Location>() {
                @Override
                public void onSuccess(Location location) {
                    Log.d("MSG","Entering updateUIValues function");
                    if(location!=null){
                        Log.d("MSG", "But Location is not empty");
                        updateUIValues(location);
                        currentLocation = location;
                    }
                    else{
                        Log.d("MSG", "But Location is empty");
                    }
                }
            });
            fusedLocationProviderClient.requestLocationUpdates(locationRequest, locationCallBack, null);
        }
        else{
            requestLocationPermission();
        }
    }

    @SuppressLint({"SimpleDateFormat", "SetTextI18n"})
    private void updateUIValues(Location location) {
        // Update the text view objects with new location

        calender = Calendar.getInstance();
        simpleDateFormat = new SimpleDateFormat("HH:mm:ss.SSS");
        DateAndTime = simpleDateFormat.format(location.getTime());

        Date ms = calender.getTime();
        Log.d("MS", String.valueOf(ms));


        tv_time.setText(DateAndTime);
        tv_accuracy.setText(String.valueOf(location.getAccuracy()));
        tv_lon.setText(String.valueOf(location.getLongitude()));
        tv_lat.setText(String.valueOf(location.getLatitude()));

        if (location.hasAltitude()){
            tv_altitude.setText(String.valueOf(location.getAltitude()));
        }
        else {
            tv_altitude.setText("Not Available");
        }
        if (location.hasSpeed()){
            tv_speed.setText(String.valueOf(location.getSpeed()));
        }
        else {
            tv_speed.setText("Not Available");
        }
        Geocoder geocoder = new Geocoder(this);
        try {
            List<Address> addresses = geocoder.getFromLocation(location.getLatitude(), location.getLongitude(), 1);
            tv_address.setText((addresses.get(0).getAddressLine(0)));
        }
        catch (Exception e){
            tv_address.setText("Unable to find Address");
        }
        MyApplication myApplication = (MyApplication)getApplicationContext();
        savedLocations = myApplication.getMyLocations();
        // show the o´number of waypoint
        tv_waypointCounts.setText(Integer.toString(savedLocations.size()));
    }
    private boolean checkStoragePermission(){
        return ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
    }
    private void requestStoragePermission(){
        requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},100);
    }
    private boolean checkLocationPermission(){
        return (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) &&
                (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED);
    }
    private void requestLocationPermission(){
        requestPermissions(new String[] {Manifest.permission.ACCESS_FINE_LOCATION}, PERMISSIONS_FINE_LOCATION);
    }
}